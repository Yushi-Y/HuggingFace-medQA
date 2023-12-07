import os
import pandas as pd


class DontPatronizeMe:
    def __init__(self, train_path, test_path):

        self.train_path = train_path
        self.test_path = test_path

        self.train_df = None
        self.test_df = None
        
        self.load_train()
        self.load_test()

    def load_train(self):
        """
        Load task 1 training set and convert the tags into binary labels.
        Paragraphs with original labels of 0 or 1 are considered to be negative examples of PCL and will have the label 0 = negative.
        Paragraphs with original labels of 2, 3 or 4 are considered to be positive examples of PCL and will have the label 1 = positive.
        It returns a pandas dataframe with paragraphs and labels.
        """
        rows = []
        with open(os.path.join(self.train_path, "dontpatronizeme_pcl.tsv")) as f:
            for line in f.readlines()[4:]:
                row = line.strip().split('\t')
                par_id, art_id, keyword, country, t, l = row
                if l == "0" or l == "1":
                    lbin = 0
                else:
                    lbin = 1
                rows.append(
                    {
                        "par_id": par_id,
                        "art_id": art_id,
                        "keyword": keyword,
                        "country": country,
                        "text": t,
                        "label": lbin,
                        "orig_label": l,
                    }
                )
        df = pd.DataFrame(
            rows,
            columns=[
                "par_id",
                "art_id",
                "keyword",
                "country",
                "text",
                "label",
                "orig_label",
            ],
        )
        self.train_df = df

    def load_test(self):
        rows = []
        with open(self.test_path) as f:
            for line in f:
                t = line.strip().split("\t")
                rows.append(t)
        self.test_df = pd.DataFrame(
            rows, columns="par_id art_id keyword country text".split()
        )
