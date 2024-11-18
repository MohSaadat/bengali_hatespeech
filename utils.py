import json
import os
import kagglehub
import glob
import subprocess
import re
import pandas as pd
from torch.utils.data import Dataset

mv_cmd = "mv" if os.name == "posix" else "move"
cp_cmd = "cp" if os.name == "posix" else "copy"

local_datasrc = "datasets/"
with open(os.path.join(local_datasrc, "bengali_metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

src_datacorpus = metadata["dataset corpus"]
bert_model = metadata["bert models"][-1]
sbert_model = metadata["sbert models"][-1]
punctuations = metadata["punctuations"]
utf_range = metadata["utf range"]

class MyDataSet(Dataset):
    def __init__(self, corpus):
        self.corpus = corpus
        dataframe = self.__cleandata(self.__getdata(corpus))
        dataframe = dataframe.sample(frac=1).reset_index(drop=True) # shuffle the rows
        self.sentences = list(dataframe.sentence)
        self.labels = list(dataframe.hate)

    def __getdata(self, datacorpus):
        datafile = os.path.join(local_datasrc, f"{datacorpus}.csv")
        if not os.path.exists(os.path.join(local_datasrc, f"{datacorpus}.csv")):
            corpus_src = src_datacorpus[datacorpus]
            cachepath = kagglehub.dataset_download(corpus_src)
            csvfiles = glob.glob(os.path.join(cachepath, "*.csv"))
            for file in csvfiles:
                if "Bengali" in file and "hate" in file and "speech" in file:
                    name_after_copied = os.path.join(local_datasrc, os.path.basename(file))
                    copynmove_cmd = f"{cp_cmd} \"{file}\" {local_datasrc} && {mv_cmd} \"{name_after_copied}\" {datafile}"
                    err = subprocess.run(copynmove_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    break

        dataframe = pd.read_csv(datafile)
        dataframe.drop(columns=['category'], inplace=True)

        return dataframe

    def __cleandata(self, dataframe):
        for sentence in dataframe.sentence:
            for punct in punctuations:
                sentence.replace(punct, '')
            sentence = re.sub(fr'[^{utf_range}\s]', '', sentence)

        return dataframe

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        label = self.labels[idx]

        return {"sentence": text, "label": label}
