import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from common import DEBUG, DEVICE

class NnBase(nn.Module):
    def update(
            self,
            data: DataLoader,
            loss_fn: nn.Module,
            nepochs: int =1,
            steps_per_epoch: int =100
    ):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=4e-3)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 2)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)
        self.train()
        for epoch_num in range(nepochs):
            loss_sum = torch.zeros(1, device=DEVICE)
            for step_num, (static, time_series, labels) in enumerate(data):
                if step_num >= steps_per_epoch:
                    break
                self.optimizer.zero_grad()
                preds = self((static, time_series))
                loss = loss_fn(preds, labels)
                loss_sum += loss
                loss.backward()
                self.optimizer.step()
            avg_loss = loss_sum / step_num
            self.scheduler.step(avg_loss)
            print(f"Epoch {epoch_num} loss: {avg_loss}")

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def eval_loss(self, dataloader, loss_fn, nbatches=1):
        preds = []
        labels = []
        for i, (static, time_series, label) in enumerate(dataloader):
            if i >= nbatches:
                break
            preds.append(self.predict((static, time_series)))
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        return loss_fn(preds, labels)

class MLP(NnBase):
    def __init__(
            self,
            nstatic_features,
            sequence_length,
            ntime_features,
            nlabels,
            hidden_size=128,
            device=DEVICE
    ):
        super().__init__()
        self.noise_std = 0.1
        input_size = sequence_length * ntime_features
        # fun notes:
        # for large models, normalization techniques such as batch normalization should be used
        # to mitigate shattering gradients (gradients that lose useful information)
        # dropout is a regularization technique that randomly drops neurons during training
        # additionally, dropout is effectively an exponential Bayesian model ensemble (Gal and Ghahramani, 2016)
        # GELU is a new non-linear activation function that is a smooth approximation of ReLU
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1), #automatically disabled in eval mode
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, nlabels),
            nn.Softmax(dim=1)
        )
        self.to(device)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]):
        static, time_series = x
        batch_size = static.shape[0]
        x = time_series.reshape(batch_size, -1)
        if self.training:
            x = x + torch.randn_like(x) * self.noise_std
        return self.model(x)


class LinearRegression(NnBase):
    def __init__(self, nfeatures, device=DEVICE):
        super().__init__()
        self.model = nn.Linear(nfeatures, 1)
        self.to(device)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def get_weights(self):
        return self.model.weight.detach().cpu().numpy().flatten()


if __name__ == "__main__":

    import nn_data
    import casas_preprocessing as cp
    df, scaler = cp.get_data()
    seq_length = 32

    # Create dataloader for each person
    dataloaders = {}
    for i, person in enumerate(df["Person"].unique()):
        dataloaders[person] = nn_data.get_dataloader(
            df[df["Person"] == person], cp.static_feats, cp.time_feats, cp.activity_cols, sequence_length=seq_length
        )

    eval_metrics = []
    startTime = datetime.now()
    for i, person1 in enumerate(dataloaders.keys()):

        # Create NN and train on one person
        mlp = MLP(
            nstatic_features=len(cp.static_feats),
            sequence_length=seq_length,
            ntime_features=len(cp.time_feats),
            nlabels=len(cp.activity_cols)
        )

        mlp.update(dataloaders[person1], nn.CrossEntropyLoss(), nepochs=25, steps_per_epoch=100)

        #Set training flag to False; change to testing on others
        mlp.eval()

        for j, person2 in enumerate(dataloaders.keys()):
            static, time_series, labels = next(iter(dataloaders[person2]))
            preds = mlp((static, time_series))
            loss = mlp.eval_loss(dataloaders[person2], nn.CrossEntropyLoss(), nbatches=100)
            mae = mlp.eval_loss(dataloaders[person2], nn.L1Loss(), nbatches=100)
            mse = mlp.eval_loss(dataloaders[person2], nn.MSELoss(), nbatches=100)
            eval_metrics.append([loss.item(), mae.item(), torch.sqrt(mse).item()])

    print(datetime.now() - startTime)

    #Save results
    n = len(dataloaders.keys())
    first_index = np.repeat(range(n),n)
    second_index = np.tile(range(n), n)
    eval_metrics_df = pd.DataFrame(data=eval_metrics, index=[first_index, second_index],
                                   columns=["Avg Loss", "MAE", "RMSE"])
    eval_metrics_df.to_csv("EvalNN_Data")
    pass
