#!usr/bin/ven python3
# -*- coding: utf-8 -*-

from datasets import *
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, \
    BertTokenizerFast, BertTokenizer, Trainer, pipeline
from sklearn.cluster import KMeans
import os
import torch


class Detector:
    def __init__(self, data_dir, max_length):
        self.data_dir = data_dir
        self.max_length = max_length

    def __datasets(self):  # prepare the custom datasets
        dataset_files = []
        for _, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if filename.endswith("token.dat"):
                    dataset_files.append(filename)

        dataset = load_dataset("text", data_dir=self.data_dir, data_files=dataset_files)
        return dataset

    def encode_with_truncation(self, examples):
        return self.load_tokenizer()(examples['text'], truncation=True, padding="max_length",
                                     max_length=self.max_length, return_special_tokens_mask=True)

    def load_tokenizer(self):
        tokenizer = BertTokenizer('./models/tokenizer-model/vocab.txt')
        special_tokens = ['[log]', '[call]', '[root]', '[start]', '[state]', '[end]', '[outs]', '[ins]']
        tokenizer.add_tokens(special_tokens)
        return tokenizer

    def load_model(self):
        model_path = "./models/tokenizer-model"
        model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-10"), output_hidden_states=True)
        return model

    def extract_features(self):
        d = self.__datasets()
        dataset = d["train"].map(self.encode_with_truncation, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        model = self.load_model()
        with torch.no_grad():
            outputs = model(dataset['input_ids'])
            embeddings = outputs.hidden_states[-1]
            cls_embeddings = embeddings[:, 0, :]
            # print(cls_embeddings)
            # print(cls_embeddings.size())
        return cls_embeddings

    def detect(self):  # detect the abnormal transaction
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(self.extract_features().numpy())

        labels = kmeans.labels_
        d = self.__datasets()
        anomaly_indices = [i for i, label in enumerate(labels) if label == 0]
        anomaly_texts = [d['train']['text'][i] for i in anomaly_indices]

        print("Anomaly texts:")
        count = 0
        for text in anomaly_texts:
            print(text)
            print('\n')
            count += 1
        print(count)

if __name__ == '__main__':

    detector = Detector(data_dir="./dataset",
                        max_length=100)
    detector.detect()


