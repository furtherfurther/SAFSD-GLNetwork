import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd


class IEMOCAPDataset(Dataset):
    """This class is used to load and process the IEMOCAP dataset, a multimodal dataset for emotion analysis that includes video, audio, text, and labels."""
    def __init__(self, train=True):
        """
        :param train: Specifies whether to load the training set or the test set.
        """
        # Load preprocessed IEMOCAP dataset features using pickle. These features include video IDs, speakers, labels, text, visual features, audio features, sentences, etc.
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('data/iemocap_multimodal_features.pkl', 'rb'), encoding='latin1')
        # Select training or test video IDs based on the value of the train parameter.
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        # Get the video ID corresponding to the current index.
        vid = self.keys[index]
        # Return different types of data including text, visual, audio features, speaker labels, video labels, and video ID.
        # These data are converted into PyTorch tensor format.
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        """This function processes data in batches during data loading. It ensures consistent shapes when creating batches."""
        # Convert the data into a pandas DataFrame.
        dat = pd.DataFrame(data)
        # Process different types of data accordingly. Text, visual, and audio features are padded using pad_sequence to ensure equal lengths.
        # Speaker and label data are converted to lists.
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]


class MELDDataset(Dataset):
    """This class is used to load and process the MELD dataset, a multimodal conversational dataset that includes text, visual, audio, and labels."""
    def __init__(self, path, train=True):
        """
        :param path: File path containing the preprocessed data.
        :param train: Specifies whether to load the training set or the test set.
        """
        # Load preprocessed data from the specified path using pickle. The data includes video IDs, speakers, labels, text, visual features, audio features, sentences, etc.
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        # Select training or test video IDs based on the value of the train parameter.
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        # Get the video ID corresponding to the current index.
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        """Return all sample labels."""
        return_label = []
        for key in self.keys:
            # Append the labels corresponding to each video ID to the list.
            return_label += self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        """This function processes data in batches during data loading."""
        # Convert the data into a pandas DataFrame.
        dat = pd.DataFrame(data)
        # Process different types of data accordingly. Text, visual, and audio features are padded using pad_sequence to ensure equal lengths.
        # Speaker features and video labels are converted to lists.
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]
