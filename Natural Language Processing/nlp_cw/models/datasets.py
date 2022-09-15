import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class DPMDataset(Dataset):
    country_mapping = {
        "us": 0,
        "pk": 1,
        "ca": 2,
        "gb": 3,
        "tz": 4,
        "lk": 5,
        "ke": 6,
        "ph": 7,
        "ng": 8,
        "jm": 9,
        "bd": 10,
        "nz": 11,
        "za": 12,
        "sg": 13,
        "in": 14,
        "gh": 15,
        "hk": 16,
        "ie": 17,
        "my": 18,
        "au": 19,
    }
    keyword_mapping = {
        "vulnerable": 0,
        "women": 1,
        "migrant": 2,
        "immigrant": 3,
        "disabled": 4,
        "in-need": 5,
        "hopeless": 6,
        "refugee": 7,
        "homeless": 8,
        "poor-families": 9,
    }

    def __init__(self, keyword, country, text, label, tokenizer, tokenizer_max_len):
        self.keyword = keyword.map(lambda x: self.keyword_mapping[x]).tolist()
        self.country = country.map(lambda x: self.country_mapping[x]).tolist()
        self.text = text.tolist()
        self.label = label.tolist()
        self.tokenizer = tokenizer
        self.tokenizer_max_len = tokenizer_max_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = self.tokenizer(
            self.text[idx],
            padding="max_length",
            max_length=self.tokenizer_max_len,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]

        categorical = torch.cat(
            [
                F.one_hot(
                    torch.tensor(self.keyword[idx]),
                    num_classes=len(self.keyword_mapping),
                ),
                F.one_hot(
                    torch.tensor(self.country[idx]),
                    num_classes=len(self.country_mapping),
                ),
            ]
        )
        return x, torch.tensor(self.label[idx]), categorical
