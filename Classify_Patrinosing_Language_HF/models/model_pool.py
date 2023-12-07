from audioop import add
from click import progressbar
from transformers import AutoModel
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

import numpy as np


class TransformerModel(pl.LightningModule):
    # one batch contains x (text tokens), y (labels), z (categorical one hot encodings)
    def __init__(
        self,
        pretrained_model_name,
        add_categorical,
        num_classes,
        lr,
        classifier_dim,
        dropout,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.name = pretrained_model_name.split("/")[-1]

        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.model.pooler = None

        # freeze layers
        # for p in self.model.parameters():
        #     p.requires_grad = False

        self.add_categorical = add_categorical

        if add_categorical:
            classifier_dim += 30

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            # nn.Linear(classifier_dim, classifier_dim),
            # nn.Dropout(dropout),
            # nn.ReLU(),
            nn.Linear(classifier_dim, num_classes),
        )
        self.lr = lr

    def loss(self, batch):
        # for training and reporting
        x, y, z = batch
        y_hat = self.model(x)["last_hidden_state"][:, 0, :]
        if self.add_categorical:
            y_hat = torch.cat([y_hat, z], dim=1)
        y_hat = self.classifier(y_hat)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def forward(self, batch):
        x, _, z = batch
        # inference
        y_hat = self.model(x)["last_hidden_state"][:, 0, :]
        if self.add_categorical:
            y_hat = torch.cat([y_hat, z], dim=1)
        y_hat = self.classifier(y_hat)
        return y_hat.argmax(dim=1)

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log("val_loss", loss)
        return loss

    # def training_epoch_end(self, outputs):
    #     print(self.optimizers().optimizer.param_groups[0]['lr'])

    def predict_step(self, batch, batch_idx):
        return self.forward(batch).tolist()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
        # lr_scueduler = ExponentialLR(optimizer, 0.9)
        # return [optimizer], [lr_scueduler]

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log("train_loss", loss)
        return loss
