import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pandas as pd
import numpy as np

from common import DEBUG, DEVICE


class StaticTimeSeriesDataset(Dataset):
    def __init__(
            self,
            df :pd.DataFrame,
            static_cols :list[str],
            time_cols :list[str],
            label_cols :list[str],
            sequence_length :int =32,
            device :str =DEVICE
    ):
        self.device = device
        def df_to_torch(df :pd.DataFrame):
            return torch.from_numpy(df.to_numpy()).to(self.device)
        self.static = df[static_cols].drop_duplicates()
        assert self.static.shape[0] == 1, "Static columns must be static"
        self.static = df_to_torch(self.static.iloc[0])

        self.time_series = df_to_torch(df[time_cols])
        self.labels = df_to_torch(df[label_cols])
        self.sequence_length = sequence_length
        # useful for debugging but not necessary and memory intensive
        self.df = df

    def __len__(self):
        return len(self.time_series) - self.sequence_length

    def __getitem__(self, idx):
        stop = idx + self.sequence_length
        return self.static, self.time_series[idx:stop], self.labels[idx:stop].mean(dim=0)



class ManyStaticTimeSeriesDataset(Dataset):
    def __init__(self, datasets :list[StaticTimeSeriesDataset]):
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)
        raise IndexError("Index out of bounds")

    def get_dataloader(self, batch_size=64, shuffle=True, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_dataloader(
        df:pd.DataFrame,
        static_cols:list[str],
        time_cols:list[str],
        label_cols:list[str],
        sequence_length:int =32,
):
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].astype(np.float32)

    datasets= []
    for name, group in df.groupby(static_cols):
        datasets.append(StaticTimeSeriesDataset(group, static_cols, time_cols, label_cols, sequence_length=sequence_length))
    return ManyStaticTimeSeriesDataset(datasets).get_dataloader()

if __name__ == "__main__":
    #note that you will have to split train and test by loading different dataloaders
    import casas_preprocessing as cp
    df, scaler = cp.get_data()
    dataloader = get_dataloader(df, cp.static_feats, cp.time_feats, cp.activity_cols)
    for i, data in enumerate(dataloader):
        static, time_series, labels = data
        print(static.shape, time_series.shape, labels.shape)
        if i > 2:
            break
    pass