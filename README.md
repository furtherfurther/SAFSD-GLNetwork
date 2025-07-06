# A Global-Local Fusion Network with Similarity-Aware Focal Self-Distillation for Multimodal Emotion Recognition in Conversations


## 🚀 Overview
Multimodal Emotion Recognition in Conversations (MERC) is widely used for mental health assistance and intelligent dialogue systems. Efficiently capturing both global semantic patterns and local structural information is crucial for MERC. However, existing sequence- and graph-based methods typically capture only one aspect, and the methods combining both remain limited in effectively modeling multimodal interactions and graph structures. Additionally, the class imbalance and limited ability in distinguishing closely related emotions also influence the accuracy and robustness of emotion recognition, which is still in the preliminary exploration stage in MERC. To overcome these challenges, this paper introduces SAFSD-GLNet, a global-local fusion network enhanced with similarity-aware focal self-distillation. First, the proposed model uses a Transformer architecture with modality-hybrid attention mechanism, along with a masked Graph Convolutional Network (GCN), to achieve comprehensive global-local multimodal fusion that captures both semantic relevance and modality complementarity. Moreover, a Similarity-Aware Focal Loss (SAF Loss) is employed to improve recognition of minority classes and enhance discrimination between similar emotional states. Finally, SAFSD-GLNet integrates self-distillation, enabling unimodal student models to learn richer representations by incorporating SAF Loss into the training objective. Experimental results on the IEMOCAP and MELD datasets demonstrate that SAFSD-GLNet surpasses existing state-of-the-art methods by 1.88\% and 1.50\% in F1-score. The code is available at https://anonymous.4open.science/r/SAFSD-GLNet.

## 🧠 SAFSD-GLNet

![SAFSD-GLNet Framework](SAFSD-GLNet.png)

## 🛠️ Setup

### Preparing the Code and Data
download them from https://anonymous.4open.science/r/SAFSD-GLNet.

### Preparing the Environment
```bash
conda create -n SAFSD-GLNet python==3.8
cd SAFSD-GLNet
conda activate SAFSD-GLNet
```
- Check the packages needed or simply run the command:
```console

pip install -r requirements.txt
```


## 🙏 Acknowledgements


