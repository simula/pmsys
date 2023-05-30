import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TimeseriesDataset(Dataset):
    """
	A dataset made for timeseries data

	Arguments:
        dataframe: dataframe of all data
        features: feature names
        target: target name(s)
        sequence_length: length of sequence/input window
	
    Returns:
		A timeseriesDataset
	"""
    def __init__(self, dataframe, features, target, sequence_length):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.X = torch.tensor(dataframe[features].values, device = ('cpu')).float()
        self.y = torch.tensor(dataframe[target].values, device = ('cpu')).float() #long/float

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


def createDataloader(train, test, batch_size, sequence_length, features, target):
    """
	Creates a dataloader for time series data

	Arguments:
        train: train data
        test: validation data
        batch_size: size of batches
        sequence_length: length of sequence/input window
        features: feature names
        target: target name(s)
	
    Returns:
		A timeseriesDataset
	"""

    torch.manual_seed(101)

    trainDataset = TimeseriesDataset(
        train,
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    testDataset = TimeseriesDataset(
        test,
        target=target,
        features=features,
        sequence_length=sequence_length
    )

    train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, trainDataset, testDataset