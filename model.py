import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_gcn import GCN

# iemocap_similarity_matrix = torch.tensor([
#     [1.0, 0.3, 0.5, 0.2, 0.8, 0.5],  # happy
#     [0.3, 1.0, 0.6, 0.7, 0.4, 0.9],  # sad
#     [0.5, 0.6, 1.0, 0.4, 0.5, 0.5],  # neutral
#     [0.2, 0.7, 0.4, 1.0, 0.3, 0.8],  # angry
#     [0.8, 0.4, 0.5, 0.3, 1.0, 0.6],  # excited
#     [0.5, 0.9, 0.5, 0.8, 0.6, 1.0]   # frustrated
# ], dtype=torch.float32)


class SimilarityAwareFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, size_average=True):
        super(SimilarityAwareFocalLoss, self).__init__()
        self.gamma = gamma
        # self.alpha = alpha
        self.alpha = torch.tensor([0.1807,0.1707,0.1585,0.1667,0.1725,0.1506]).cuda()  # Set according to class frequencies
        self.size_average = size_average

    def forward(self, logits, labels, similarity_matrix=iemocap_similarity_matrix):
        """
        :param logits: Model predictions
        :param labels: Ground truth labels
        :param similarity_matrix: Similarity matrix between classes
        """
        similarity_matrix = similarity_matrix.to(labels.device)
        labels = labels.view(-1)
        label_onehot = torch.zeros(logits.size(0), logits.size(1)).to(logits.device).scatter_(1, labels.unsqueeze(1), 1)

        log_p = F.log_softmax(logits, dim=-1)
        pt = torch.exp(log_p)
        sub_pt = 1 - pt

        # Similarity-based weighting
        similarity_weights = similarity_matrix[labels]  # Get weights from similarity matrix for each label

        # Compute Focal Loss with similarity weights
        focal_loss = -self.alpha * (sub_pt ** self.gamma) * log_p * similarity_weights.unsqueeze(1)

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class KLDivLoss(nn.Module):
    """KL Divergence Loss without mask, normalized by batch size"""
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, log_pred, target):
        """
        :param log_pred: Log probabilities output by model
        :param target: Target distribution
        """
        loss = self.loss(log_pred, target) / log_pred.size(0)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        """
        Calculates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits, -1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


def gelu(x):
    """
    Implements the Gaussian Error Linear Unit (GELU) activation function.
    GELU is popular in deep learning, especially NLP tasks.
    It can be seen as a compromise between ReLU and Sigmoid/Tanh,
    combining ReLU's non-saturation and Sigmoid/Tanh's smoothness.
    Alternatively, you can directly call nn.GELU().
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network (FFN) in Transformer models.
    This FFN enhances the model's nonlinear processing capability via two linear layers and an activation.
    Residual connections and layer normalization help prevent gradient vanishing during training.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model: Feature dimension of the model
        :param d_ff: Hidden dimension of the feed-forward network
        :param dropout: Dropout rate
        """
        super(PositionwiseFeedForward, self).__init__()
        # Linear layer projecting from d_model to d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        # Linear layer projecting from d_ff back to d_model
        self.w_2 = nn.Linear(d_ff, d_model)
        # Layer normalization to normalize input features
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # GELU activation function
        self.actv = gelu
        # Dropout layers to prevent overfitting during training
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)



class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention mechanism, a key component in Transformer.
    It allows the model to learn information in parallel across different representation subspaces.
    """
    def __init__(self, head_count, model_dim, dropout=0.1):
        """
        :param head_count: Number of attention heads
        :param model_dim: Model feature dimension
        """
        # Ensure model_dim is divisible by head_count as required by multi-head attention
        assert model_dim % head_count == 0
        # Compute dimension per head


    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """Reshape for multi-head projection"""
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """Reshape back after multi-head attention"""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # Apply linear transformations and reshape for multi-head attention
        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        # Scale query by sqrt(dim_per_head)
        query = query / math.sqrt(dim_per_head)
        # Calculate attention scores via dot product of query and key
        scores = torch.matmul(query, key.transpose(2, 3))




class PositionalEncoding(nn.Module):
    """
    Positional Encoding, a Transformer component,
    which adds sine and cosine values of positions to the input sequence,
    giving the model information about word positions in the sequence.
    """
    def __init__(self, dim, max_len=512):
        """
        :param dim: Dimension of positional encoding
        :param max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        # Initialize positional encoding tensor
        pe = torch.zeros(max_len, dim)
        # Positions 0 to max_len-1
        position = torch.arange(0, max_len).unsqueeze(1)
        # Divisor term for scaling in sine/cosine
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        # Calculate positional encodings for even indices with sine
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        # For odd indices with cosine
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        # Add batch dimension for broadcasting
        pe = pe.unsqueeze(0)
        # Register as buffer so it's saved but not trained
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        # Add positional encoding and speaker embedding to input
        x = x + pos_emb + speaker_emb
        return x


class TransformerEncoderLayer(nn.Module):
    """This class implements a Transformer encoder layer, which consists of self-attention
    and a feed-forward network, and includes layer normalization and residual connections."""
    def __init__(self, d_model, heads, d_ff, dropout):
        """
        :param d_model: Feature dimension of the model
        :param heads: Number of heads in multi-head attention
        :param d_ff: Hidden dimension of the feed-forward network
        :param dropout: Dropout rate
        """
        super(TransformerEncoderLayer, self).__init__()
        # Implements self-attention mechanism
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        # Implements feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # Normalizes input features
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        # Checks if inputs_a and inputs_b are equal. If yes, they represent different
        # views of the same input, typically used in Transformer decoders where inputs_b
        # is the encoder output
        if inputs_a.equal(inputs_b):
            # If not the first iteration (usually in incremental training), apply layer norm to inputs_b
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            # Adjust mask shape to match self-attention requirements
            mask = mask.unsqueeze(1)
            # Apply self-attention to compute context
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            # If inputs_a and inputs_b differ, typically in encoder part
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)

        # Apply dropout and add residual connection
        out = self.dropout(context) + inputs_b
        # Pass residual output through feed-forward network and return
        return self.feed_forward(out)


    def forward(self, x_a, x_b, mask, speaker_emb):
        # Check if x_a and x_b are equal. If yes, they represent different views of same input,
        # typical in Transformer decoders where x_b is encoder output
        if x_a.equal(x_b):
            # Add positional encoding and speaker embedding to x_b
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            # Pass through all encoder layers
            for i in range(self.layers):
                # For each encoder layer, pass x_b as input with mask to filter irrelevant info
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        else:
            # If x_a and x_b differ, typically in encoder part
            # Add positional encoding and speaker embedding to both x_a and x_b
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                # For each encoder layer, pass both x_a and x_b as inputs with mask
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b
da:0")
    return node_features


class Transformer_Based_Model(nn.Module):
    """This class implements a Transformer-based multimodal emotion classification model.
    It processes text, visual, and audio data and predicts emotion categories."""
    """It also incorporates knowledge distillation within the model."""
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout, D_g=1024, graph_hidden_size=1024, num_L=3, num_K=4, modals='avl'):
        """
        :param dataset: Name or type of dataset used. The model may adjust fusion or processing based on dataset.
        :param temp: Temperature parameter, commonly used to smooth softmax outputs.
        :param D_text: Input text feature dimension.
        :param D_visual: Input visual feature dimension.
        :param D_audio: Input audio feature dimension.
        :param n_classes: Number of emotion categories to predict.
        :param hidden_dim: Hidden dimension size of the model, used for Transformer encoders, conv layers, embeddings, etc.
        :param n_speakers: Number of speakers in the dialogue, used for speaker embedding size and speaker info processing.
        """
        super(Transformer_Based_Model, self).__init__()
        # Save temperature parameter, typically used to smooth softmax output
        self.temp = temp
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        self.dropout = dropout
        """
         2. Usage of n_speakers: speaker mask and index adjustments
         If n_speakers is 2 or 9, different padding_idx and padding strategies are set.
        """
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        # Create an embedding layer to convert speaker indices to embeddings
        """
        1. Usage of n_speakers: embedding layer size
        n_speakers defines the size of speaker embedding layer (self.speaker_embeddings).
        The input indices range from 0 to n_speakers, and the embedding outputs vectors for each speaker.
        The +1 is for padding index reserved in nn.Embedding, specified by padding_idx.
        """
        self.speaker_embeddings = nn.Embedding(n_speakers + 1, hidden_dim, padding_idx)

        # Temporal convolutional layers
        # Three 1D conv layers transform input features of different modalities to model hidden dimension
        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)


        # Three linear layers to reduce dimensions of fused or concatenated unimodal features
        self.features_reduce_t = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_a = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_v = nn.Linear(3 * hidden_dim, hidden_dim)

        # Multimodal-level Gated Fusion for fusion across modalities
        self.last_gate = Multimodal_GatedFusion(hidden_dim)

        # Emotion Classifier
        # Three sequential models for emotion classification per modality
        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

        # Graph convolutional model (GCN)
        self.graph_model = GCN(n_dim=D_g, nhidden=graph_hidden_size,
                               dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True,
                               return_feature=True, use_residue=False, n_speakers=n_speakers,
                               modals='avl', use_speaker=True, use_modal=False, num_L=num_L,
                               num_K=num_K)
        self.multi_modal = True
        self.att_type = 'concat_DHT'
        self.modals = [x for x in modals]  # a, v, l
        self.use_residue = False

        self.all_output_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, epoch=None):
        # Obtain speaker indices at each timestep using qmask (indicating speaker changes)
        # qmask shape changed for compatibility with this function (changed from train_or_eval_model)
        spk_idx = torch.argmax(qmask, -1)
        spk_idx = torch.argmax(qmask.permute(1, 0, 2), -1)
        origin_spk_idx = spk_idx

        if self.n_speakers == 2:
            # dia_len contains lengths of each dialogue, i is dialogue index, x is its length
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        # Convert speaker indices to embeddings
        spk_embeddings = self.speaker_embeddings(spk_idx)

        # Temporal convolutional layers
        # Process input features through 1D conv layers, converting to model hidden dim
        textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)

        # Intra- and Inter-modal Transformers: encode modality features
        # The following three variables represent "intra" Transformer encoder outputs for text, audio, visual modalities
        t_t_transformer_out = self.t_t(textf, textf, u_mask, spk_embeddings)
        a_t_transformer_out = self.a_t(acouf, textf, u_mask, spk_embeddings)
        v_t_transformer_out = self.v_t(visuf, textf, u_mask, spk_embeddings)

        # Inter-modal Transformer encoder outputs
        a_a_transformer_out = self.a_a(acouf, acouf, u_mask, spk_embeddings)
        t_a_transformer_out = self.t_a(textf, acouf, u_mask, spk_embeddings)
        v_a_transformer_out = self.v_a(visuf, acouf, u_mask, spk_embeddings)

        # Interactions between visual-text, audio-text, audio-visual modalities
        v_v_transformer_out = self.v_v(visuf, visuf, u_mask, spk_embeddings)
        t_v_transformer_out = self.t_v(textf, visuf, u_mask, spk_embeddings)
        a_v_transformer_out = self.a_v(acouf, visuf, u_mask, spk_embeddings)



        # Gated fusion of Transformer encoder outputs from different modalities
        t_transformer_out = self.features_reduce_t(torch.cat([t_t_transformer_out, a_t_transformer_out, v_t_transformer_out], dim=-1))
        a_transformer_out = self.features_reduce_a(torch.cat([a_a_transformer_out, t_a_transformer_out, v_a_transformer_out], dim=-1))
        v_transformer_out = self.features_reduce_v(torch.cat([v_v_transformer_out, t_v_transformer_out, a_v_transformer_out], dim=-1))



        # Graph model computes final multimodal features
        emotions_feat = self.graph_model(features_a, features_v, features_l, dia_len, qmask, epoch)
        emotions_feat = self.dropout_(emotions_feat)
        emotions_feat = nn.ReLU()(emotions_feat)

        # Log softmax and softmax for emotion classification
        log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
        prob = F.softmax(self.smax_fc(emotions_feat), 1)

        # Multimodal-level Gated Fusion of all modality features
        all_transformer_out = self.last_gate(t_transformer_out, a_transformer_out, v_transformer_out)

        # Emotion classifiers: get prediction outputs for each modality and combined modalities
        t_final_out = self.t_output_layer(t_transformer_out)
        a_final_out = self.a_output_layer(a_transformer_out)
        v_final_out = self.v_output_layer(v_transformer_out)
        all_final_out = self.all_output_layer(all_transformer_out)
        t_final_out_1 = self.t_output_layer(features_l)
        a_final_out_1 = self.a_output_layer(features_a)
        v_final_out_1 = self.v_output_layer(features_v)

        all_log_prob = F.log_softmax(all_final_out, 2)
        all_prob = F.softmax(all_final_out, 2)



        kl_all_prob = F.softmax(all_final_out / self.temp, 2)
        kl_all_prob_1 = F.softmax(self.smax_fc(emotions_feat) / self.temp, 1)

        # Return log probabilities, probabilities, and temperature-scaled outputs for different modalities and fusion
        return (t_log_prob, a_log_prob, v_log_prob, log_prob, prob,
                kl_t_log_prob_1, kl_a_log_prob_1, kl_v_log_prob_1, kl_all_prob_1)
