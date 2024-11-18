import os
import kagglehub
import glob
import subprocess
import pandas as pd
import re
import collections
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import time

samples_per_label = 50

mv_cmd = "mv" if os.name == "posix" else "move"
cp_cmd = "cp" if os.name == "posix" else "copy"

local_datasrc = os.path.join("..", "datasets")
with open(os.path.join(local_datasrc, "bengali_metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

src_datacorpus = metadata["dataset corpus"]
bert_model = metadata["bert models"][-1]
sbert_model = metadata["sbert models"][-1]
punctuations = metadata["punctuations"]
utf_range = metadata["utf range"]

def getdata(datacorpus):
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

def cleandata(dataframe):
    for sentence in dataframe.sentence:
        for punct in punctuations:
            sentence.replace(punct, '')
        sentence = re.sub(fr'[^{utf_range}\s]', '', sentence)

    return dataframe

def show_2d_embedding(datacorpus, use_sentence_transformer=True, use_tsne=True):
    # get the data
    dataframe = cleandata(getdata(datacorpus))
    dataframe = dataframe.sample(frac=1).reset_index(drop=True) # shuffle the rows
    sentences = list(dataframe.sentence)
    labels = list(dataframe.hate)

    # reduce data
    ordinary = [i for i, x in enumerate(labels) if x == 0]
    hateful = [i for i, x in enumerate(labels) if x == 1]
    keep_idx = ordinary[:samples_per_label]
    keep_idx.extend(hateful[:samples_per_label])
    sentences = [sentences[i] for i in keep_idx]
    labels = [labels[i] for i in keep_idx]

    label_counts = collections.Counter(labels)
    for key in label_counts.keys():
        if key not in [0, 1]:
            print(f"Unexpected label found")
            return 1
    print(f"\nNumber of hate sentences: {label_counts[0]}")
    print(f"Number of ordinary sentences: {label_counts[1]}\n")

    # first, embed the sentences to high dimensional vector
    # next, reduce the high-dimensional embeddings to 2d

    if use_sentence_transformer:
        model = SentenceTransformer(sbert_model)
        sentence_embeddings_high = model.encode(sentences)
    else:
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        model = AutoModel.from_pretrained(bert_model)

        sentence_tokens = tokenizer(
                                    sentences,
                                    max_length=model.config.max_position_embeddings,
                                    padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            sentence_embeddings_high = model(**sentence_tokens).last_hidden_state.mean(dim=1)

    print('Sentence embedding complete')

    if use_tsne:
        tsne = TSNE(n_components=2, random_state=42)
        sentence_embeddings_2d = tsne.fit_transform(sentence_embeddings_high)
    else:
        scaler = StandardScaler()
        reducer = umap.UMAP(n_components=2, n_neighbors=int(samples_per_label/10), min_dist=0.1, random_state=42)
        sentence_embeddings_high_scaled = scaler.fit_transform(sentence_embeddings_high)
        sentence_embeddings_2d = reducer.fit_transform(sentence_embeddings_high_scaled)

    print('Dimensionality reduction complete')

    # visualization
    plt.figure(figsize=(12, 8))
    for label in [0, 1]:
        label2show = "ordinary" if label == 0 else "hateful"
        plt.scatter(
            sentence_embeddings_2d[np.array(labels) == label, 0],
            sentence_embeddings_2d[np.array(labels) == label, 1],
            label=f'{label2show}',
            alpha=0.7,
            s=50
        )

    plt.title('2D Embedding')
    plt.xlabel('Comp 1')
    plt.ylabel('Comp 2')
    plt.legend()
    plt.savefig('2d_embedding.png')
    plt.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--use_tokenizer', action='store_true', help='Use this flag to use tokenizer and bert model')
    parser.add_argument('-t', '--use_umap', action='store_true', help='Use this flag to use umap')
    args = parser.parse_args()

    t_start = time.time()
    show_2d_embedding('bhs_naurosromim', use_sentence_transformer=not args.use_tokenizer, use_tsne=not args.use_umap)
    print(f"Time elapsed: {time.time()-t_start}")
