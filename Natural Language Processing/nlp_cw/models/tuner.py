import pandas as pd
from collections import Counter
from dont_patronize_me import DontPatronizeMe
from models.datasets import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
from itertools import chain

from sklearn.metrics import precision_score, recall_score, f1_score
import gc


class ModelTuner:
    def __init__(self, target_label):
        self.target_label = target_label
        # (label > threshold) -> 1, (label <= threshold) -> 0, constant should not be changed!
        self.threshold = 1

    # "target_label" is the true target field
    # "label" or "orig_label"
    def load_data(self):
        dpm = DontPatronizeMe("./data", "./data/task4_test.tsv")
        full_df = dpm.train_df
        full_df["target_label"] = dpm.train_df[self.target_label].astype(int)
        test_df = dpm.test_df
        return full_df, test_df

    @classmethod
    def df_to_dataset(cls, df, tokenizer, tokenizer_max_len):
        return DPMDataset(
            df["keyword"],
            df["country"],
            df["text"],
            df["target_label"],
            tokenizer,
            tokenizer_max_len
        )

    def downsample_data(self, df):
        result = pd.DataFrame()
        freq = Counter(df["target_label"])
        min_freq = min(freq.values())
        for l in freq.keys():
            result = pd.concat(
                [result, df[df["target_label"] == l].sample(n=min_freq, replace=False)]
            )
        return result

    def get_metrics(self, trainer, dataloader):
        y_hat = np.array(
            list(chain(*trainer.predict(dataloaders=dataloader, ckpt_path='best')))
        )
        y_true = np.array(list(chain(*[y.tolist() for _, y, _ in dataloader])))
        if self.target_label == "orig_label":
            y_hat = (y_hat > self.threshold).astype(int)
            y_true = (y_true > self.threshold).astype(int)
        metrics = {
            "precision": precision_score(y_true, y_hat),
            "recall": recall_score(y_true, y_hat),
            "f1": f1_score(y_true, y_hat),
        }
        return metrics

    def train_once(
        self,
        train_df,
        epochs,
        patience,
        tokenizer,
        model,
        batch_size,
        gradient_clip_val,
        tokenizer_max_len,
        val_df,
    ):
        train_dataset = self.df_to_dataset(train_df, tokenizer, tokenizer_max_len)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        val_dataset = self.df_to_dataset(val_df, tokenizer, tokenizer_max_len)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience),
            ModelCheckpoint(monitor="val_loss"),
        ]
        
        logger = TensorBoardLogger(save_dir='lightning_logs', version=model.name)
        

        trainer = pl.Trainer(
            log_every_n_steps=1,
            max_epochs=epochs,
            gpus=1,
            precision=16,
            callbacks=callbacks,
            gradient_clip_val=gradient_clip_val,
            logger=logger
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        
        metrics = self.get_metrics(trainer, val_dataloader)
        return trainer, metrics
