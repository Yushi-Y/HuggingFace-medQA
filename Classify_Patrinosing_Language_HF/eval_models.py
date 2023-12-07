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
import os

from torch.utils.data import Dataset, DataLoader, Subset
from models.utils import *

target_label = "label"
tokenizer_max_len = 128
batch_size = 32

tuner = ModelTuner(target_label)
full_df, test_df = tuner.load_data()

train_idx = (pd.read_csv("data/train_semeval_parids-labels.csv")["par_id"].astype(int) - 1)
val_idx = pd.read_csv("data/dev_semeval_parids-labels.csv")["par_id"].astype(int) - 1

train_df = full_df.iloc[train_idx]
val_df = full_df.iloc[val_idx]

# pretrained_model_name_ls = [
#     "distilbert-base-uncased",
#     "nlptown/bert-base-multilingual-uncased-sentiment",
#     "roberta-base",
# ]
pretrained_model_name_ls = ['distilbert-base-uncased']

for pretrained_model_name in pretrained_model_name_ls:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    val_dataset = tuner.df_to_dataset(val_df, tokenizer, tokenizer_max_len)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    path = f"lightning_logs/default/{pretrained_model_name.split('/')[-1]}/checkpoints"
    ckpt_name = os.listdir(path)[0]

    model = TransformerModel.load_from_checkpoint(f"{path}/{ckpt_name}")
    model.eval()

    trainer = pl.Trainer(gpus=1, precision=16, enable_checkpointing=False)
    y_true = np.array(list(chain(*[y.tolist() for _, y, _ in val_dataloader])))
    y_hat = np.array(list(chain(*trainer.predict(model, val_dataloader))))

    print(f"{pretrained_model_name}\n{classification_report(y_true, y_hat)}")

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
