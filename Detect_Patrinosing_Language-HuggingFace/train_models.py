# this file will ensemble several models trained on the best hyperparameters found

from transformers import AutoTokenizer
import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd
import torch
from collections import Counter
from sklearn.metrics import classification_report
import numpy as np
from dont_patronize_me import DontPatronizeMe
from sklearn.model_selection import KFold
from models.tuner import *
from models.model_pool import *
import gc
from models.utils import *
from torch.utils.data import Dataset, DataLoader, Subset
from data_processing import *

pl.seed_everything(1234)

epochs = 20
patience = 3

target_label = "label"
num_classes = 2

categorical = False
lr = 1e-4
dropout = 0.3

batch_size = 32
gradient_clip_val = 100
tokenizer_max_len = 128

tuner = ModelTuner(target_label)
full_df, test_df = tuner.load_data()

train_idx = (pd.read_csv("data/train_semeval_parids-labels.csv")["par_id"].astype(int) - 1)
val_idx = pd.read_csv("data/dev_semeval_parids-labels.csv")["par_id"].astype(int) - 1

train_df = full_df.iloc[train_idx]
train_df = augment_data(train_df, 2)

val_df = full_df.iloc[val_idx]

train_df = tuner.downsample_data(train_df)

pretrained_model_name_ls = [
    ["distilbert-base-uncased", 768],
    ["nlptown/bert-base-multilingual-uncased-sentiment", 768],
    ["roberta-base", 768],
]
# pretrained_model_name_ls = [["distilbert-base-uncased", 768]]

for pretrained_model_name, classifier_dim in pretrained_model_name_ls:

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    model = TransformerModel(
        pretrained_model_name,
        add_categorical=categorical,
        num_classes=num_classes,
        lr=lr,
        classifier_dim=classifier_dim,
        dropout=dropout,
    )
    trainer, metrics = tuner.train_once(
        train_df,
        epochs=epochs,
        patience=patience,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        gradient_clip_val=gradient_clip_val,
        tokenizer_max_len=tokenizer_max_len,
        val_df=val_df,
    )

    print(metrics)

    test_df["target_label"] = -1
    test_dataset = tuner.df_to_dataset(test_df, tokenizer, tokenizer_max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    test_predictions = trainer.predict(dataloaders=test_dataloader, ckpt_path="best")

    labels2file(
        np.array(test_predictions).flatten(),
        f"predictions/{pretrained_model_name.split('/')[-1]}_pred.txt",
    )

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

majority_vote("predictions")
