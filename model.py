import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import json

local_datasrc = "datasets/"
with open(os.path.join(local_datasrc, "bengali_metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

class STransformer_backbone(nn.Module):
    def __init__(self, model_name):
        super(STransformer_backbone, self).__init__()
        self.model_name = model_name
        self.stransformer = SentenceTransformer(model_name)
        for param in self.stransformer:
            param.requires_grad = False

    def __getitem__(self, idx):
        return self.stransformer[idx]

    def forward(self, sentences):
        sentence_embeddings = self.stransformer.encode(sentences,
                                                        convert_to_tensor=True,
                                                        convert_to_numpy=False,
                                                        device=self.stransformer.device)
        return sentence_embeddings

class Transformer_backbone(nn.Module):
    def __init__(self, model_name):
        super(Transformer_backbone, self).__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        for param in self.transformer:
            param.requires_grad = False

    def forward(self, sentences):
        sentence_tokens = tokenizer(
                                    sentences,
                                    max_length=self.transformer.config.max_position_embeddings,
                                    padding=True, truncation=True, return_tensors="pt")
        sentence_embeddings = self.transformer(**sentence_tokens)
        return sentence_embeddings

class Classifier(nn.Module):
    def __init__(self, backbone_model_name, device, is_training=False):
        super(Classifier, self).__init__()

        self.is_stransformer = True if backbone_model_name in metadata["sbert models"] else False
        if self.is_stransformer:
            assert (backbone_model_name not in metadata["bert models"]), "How is the backbone model name in lists of both sbert models and bert models"
            self.backbone = STransformer_backbone(backbone_model_name)
        else:
            assert (backbone_model_name in metadata["bert models"]), "Unrecognized bert model"
            self.backbone = Transformer_backbone(backbone_model_name)

        self.embedding_dimension = self.__get_embedding_dim()

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dimension, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3,),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )

        self.set_trainable(is_training)
        self.set_device(device)

    def __get_embedding_dim(self):
        if self.is_stransformer:
            return self.backbone[0].get_word_embedding_dimension()
        else:
            return self.backbone.transformer.config.hidden_size

    def set_trainable(self, is_training):
        self.is_training = is_training
        for layer in self.classifier:
            if isinstance(layer, nn.Module):
                for param in layer.parameters():
                    param.requires_grad = is_training

    def get_all_trainables(self):
        trainables = []
        for p in self.parameters():
            if p.requires_grad:
                trainables.append(p)

        return trainables

    def set_device(self, device):
        available_devices = ['cuda:'+str(idx) for idx in range(torch.cuda.device_count())]
        available_devices.append('cpu')
        assert (
            (isinstance(device, int) and device in range(-1,torch.cuda.device_count())) or
            (isinstance(device, str) and device in available_devices) or
            (isinstance(device, torch.device) and (device.type == 'cpu' or (device.type == 'cuda' and device.index in range(torch.cuda.device_count()))))
            ), "Unavailable device or unrecognized device format"

        if isinstance(device, int):
            device = 'cpu' if device == -1 else f'cuda:{device}'
        self.device = device
        self.to(device)

    def forward(self, sentences):
        '''
            sentences is a list of strings
        '''
        sentence_embeddings = self.backbone(sentences)
        return self.classifier(sentence_embeddings)



