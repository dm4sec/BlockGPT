#!usr/bin/ven python3
# -*- coding: utf-8 -*-

import argparse
import logging, sys
import time
from collections import OrderedDict

from datasets import load_dataset
from web3 import Web3
from pymongo import MongoClient, ASCENDING

logging.basicConfig(stream=sys.stdout, format="%(levelname)s: %(asctime)s: %(message)s", level=logging.INFO,
                    datefmt='%a %d %b %Y %H:%M:%S')
log = logging.getLogger(__name__)
import faulthandler
faulthandler.enable()
from tokenizers import BertWordPieceTokenizer
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, BertTokenizerFast, Trainer

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm

from reformer_pytorch import Reformer, ReformerLM
from transformers import BertTokenizer, PreTrainedTokenizer
from fairseq.optim.adafactor import Adafactor

import json
import logging
from datetime import datetime

import os
import json

from config import MODEL_CONFIG
MAX_SEQ_LEN = 512
# MAX_SEQ_LEN = 128

class TxtDataset(Dataset):

    def __init__(self, file_path):
        self.documents = []
        for dirpath, _, filenames in os.walk(args.train_classifier):
            for filename in filenames:
                if filename.endswith("token.dat"):
                    with open(os.path.join(dirpath, filename)) as frh:
                        raw_text = frh.read().splitlines()
                        for line in raw_text:
                            line = re.sub('\\s+', ' ', line).strip()
                            self.documents.append([line])

    def __len__(self):
        """ Returns the number of documents. """
        return len(self.documents)

    def __getitem__(self, idx):
        document = self.documents[idx]
        return [document]

class ReformerTrainer(object):

    def __init__(self,
                 dataset,
                 model,
                 tokenizer,
                 device=None,
                 train_batch_size=8,
                 eval_batch_size=None,
                 tb_writer=False,
                 tb_dir='./models/tb_logs',
                 log_dir='./models/logs'):
        """
        Provides an easy to use class for pretraining and evaluating a Reformer Model.

        :param dataset: (torch.utils.data.Dataset) containing all of the data you wish to utilize during training.
        :param model: (reformer_pytorch.Reformer)
        :param tokenizer: (transformers.PreTrainedTokenizer) defaults to BertTokenizer ('bert-base-case')
        :param device: provide manual device placement. If None, will default to cuda:0 if available.
        :param tb_writer: (bool) Whether to write to tensorboard or not.
        :param tb_dir: (str) Where to write TB logs to.
        :param log_dir: (str) Where to write generic logs to.
        """

        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_gpu = torch.cuda.device_count()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tb_writer = tb_writer
        self.log_dir = log_dir

        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        if device is None:
            self.device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'

        if eval_batch_size is None:
            self.eval_batch_size = train_batch_size

        if tb_writer:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=tb_dir)

        logging.basicConfig(filename=f'{log_dir}/{datetime.now().date()}.log', level=logging.INFO)

    def build_dataloaders(self, train_test_split=0.1, train_shuffle=True, eval_shuffle=True):
        """
        Builds the Training and Eval DataLoaders

        :param train_test_split: The ratio split of test to train data.
        :param train_shuffle: (bool) True if you wish to shuffle the train_dataset.
        :param eval_shuffle: (bool) True if you wish to shuffle the eval_dataset.
        :return: train dataloader and evaluation dataloader.
        """
        dataset_len = len(self.dataset)
        eval_len = int(dataset_len * train_test_split)
        train_len = dataset_len - eval_len
        train_dataset, eval_dataset = random_split(self.dataset, (train_len, eval_len))
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=train_shuffle)
        eval_loader = DataLoader(eval_dataset, batch_size=self.eval_batch_size, shuffle=eval_shuffle)
        logging.info(f'''train_dataloader size: {len(train_loader.dataset)} | shuffle: {train_shuffle}
                         eval_dataloader size: {len(eval_loader.dataset)} | shuffle: {eval_shuffle}''')
        return train_loader, eval_loader

    def mask_tokens(self, inputs: torch.Tensor, mlm_probability=0.15, pad=True):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = inputs.clone()
        # mlm_probability is set to 0.15 by defaults in Bert
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # we replace masked input tokens with tokenizer.mask_token ([MASK]) by chance of 80%
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # we replace masked input tokens with random word by chance of 10%
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        if pad:
            input_pads = MAX_SEQ_LEN - inputs.shape[-1]
            label_pads = MAX_SEQ_LEN - labels.shape[-1]

            inputs = F.pad(inputs, pad=(0, input_pads), value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, pad=(0, label_pads), value=self.tokenizer.pad_token_id)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def _tokenize_input_ids(self, input_ids: list, pad_to_max_length: bool = True):
        """
        Helper function to clean up the train and eval functions
        :param input_ids: inputs to tokenize.
        :param pad_to_max_length: Whether you want to pad the inputs to the tokenizer.max_len
        :return: Tensor containing training data.
        """
        inputs = torch.cat(
            [
                self.tokenizer.encode(
                    input_ids[i],
                    add_special_tokens=True,
                    max_length=MAX_SEQ_LEN,
                    truncation=True,
                    # padding=pad_to_max_length,
                    padding="max_length",
                    return_tensors='pt'
                ) \
                for i in range(len(input_ids))
            ]
        )
        return inputs

    def train(self,
              epochs,
              train_dataloader,
              eval_dataloader,
              log_steps,
              ckpt_steps,
              ckpt_dir=None,
              gradient_accumulation_steps=1):
        """
        Trains the Reformer Model
        :param epochs: The number of times you wish to loop through the dataset.
        :param train_dataloader: (torch.utils.data.DataLoader) The data to train on.
        :param eval_dataloader: (torch.utils.data.DataLoader) The data to evaluate on.
        :param log_steps: The number of steps to iterate before logging.
        :param ckpt_steps: The number of steps to iterate before checkpointing.
        :param ckpt_dir: The directory to save the checkpoints to.
        :param gradient_accumulation_steps: Optional gradient accumulation.
        :return: Total number of steps, total loss, model
        """

        optimizer = Adafactor(self.model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        losses = {}
        global_steps = 0
        local_steps = 0
        step_loss = 0.0

        if ckpt_dir is not None:
            # assert os.path.isdir(ckpt_dir)
            if not os.path.isdir(ckpt_dir):
                os.mkdir(ckpt_dir)

            try:
                logging.info(f'{datetime.now()} | Continuing from checkpoint...')
                self.model.load_state_dict(torch.load(f'{ckpt_dir}/model_state_dict.pt', map_location=self.device))
                optimizer.load_state_dict(torch.load(f'{ckpt_dir}/optimizer_state_dict.pt'))

            except Exception as e:
                logging.info(f'{datetime.now()} | No checkpoint was found | {e}')

        self.model.train()

        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
            logging.info(f'{datetime.now()} | Utilizing {self.n_gpu} GPUs')

        self.model.to(self.device)
        logging.info(f'{datetime.now()} | Moved model to: {self.device}')
        logging.info(
            f'{datetime.now()} | train_batch_size: {self.train_batch_size} | eval_batch_size: {self.eval_batch_size}')
        logging.info(f'{datetime.now()} | Epochs: {epochs} | log_steps: {log_steps} | ckpt_steps: {ckpt_steps}')
        logging.info(f'{datetime.now()} | gradient_accumulation_steps: {gradient_accumulation_steps}')

        min_loss = 0
        for epoch in tqdm(range(epochs), desc='Epochs', position=0):
            logging.info(f'{datetime.now()} | Epoch: {epoch}')
            for step, batch in tqdm(enumerate(train_dataloader),
                                    desc='Epoch Iterator',
                                    position=1,
                                    leave=True,
                                    total=len(train_dataloader)):
                for data in batch:
                    inputs = self._tokenize_input_ids(data, pad_to_max_length=True)
                    inputs, labels = self.mask_tokens(inputs)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    output = self.model(inputs)

                    # only calculating loss on masked tokens
                    loss_mx = labels != -100
                    output = output[loss_mx].view(-1, self.tokenizer.vocab_size)
                    labels = labels[loss_mx].view(-1)

                    loss = loss_fn(output, labels)

                    if gradient_accumulation_steps > 1:
                        loss /= gradient_accumulation_steps

                    loss.backward()

                    step_loss += loss.item()
                    losses[global_steps] = loss.item()
                    local_steps += 1
                    global_steps += 1

                    if global_steps % gradient_accumulation_steps == 0:
                        optimizer.step()
                        self.model.zero_grad()

                    if global_steps % log_steps == 0:
                        if self.tb_writer:
                            self.writer.add_scalar('Train/Loss', step_loss / local_steps, global_steps)
                            self.writer.close()
                        logging.info(
                            f'''{datetime.now()} | Train Loss: {step_loss / local_steps} | Steps: {global_steps}''')

                        with open(f'{self.log_dir}/train_results.json', 'w') as results_file:
                            json.dump(losses, results_file)
                        step_loss = 0.0
                        local_steps = 0

                    if global_steps % ckpt_steps == 0:
                        # evaluating before every checkpoint
                        self.evaluate(eval_dataloader)
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        torch.save(model_to_save.state_dict(), f'{ckpt_dir}/model_state_dict.pt')
                        torch.save(optimizer.state_dict(), f'{ckpt_dir}/optimizer_state_dict.pt')

                        logging.info(f'{datetime.now()} | Saved checkpoint to: {ckpt_dir}')

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), f'{ckpt_dir}/model_state_dict.pt')
        torch.save(optimizer.state_dict(), f'{ckpt_dir}/optimizer_state_dict.pt')

        return self.model

    def evaluate(self, dataloader):
        """
        Runs through the provided dataloader with torch.no_grad()
        :param dataloader: (torch.utils.data.DataLoader) Evaluation DataLoader
        :return: None
        """
        loss_fn = nn.CrossEntropyLoss()

        if self.n_gpu > 1 and not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)

        self.model.eval()
        eval_loss = 0.0
        perplexity = 0.0
        eval_steps = 0

        logging.info(f'{datetime.now()} | Evaluating...')
        for step, batch in tqdm(enumerate(dataloader), desc='Evaluating', leave=True, total=len(dataloader)):
            for data in batch:
                inputs = self._tokenize_input_ids(data, pad_to_max_length=True)
                # print(inputs)
                # print(len(inputs))
                inputs, labels = self.mask_tokens(inputs)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    output = self.model(inputs)

                loss_mx = labels != -100
                output_ids = output[loss_mx].view(-1, self.tokenizer.vocab_size)
                labels = labels[loss_mx].view(-1)
                tmp_eval_loss = loss_fn(output_ids, labels)
                tmp_perplexity = torch.exp(tmp_eval_loss)

                if self.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()

                eval_loss += tmp_eval_loss.item()
                perplexity += tmp_perplexity.item()
                eval_steps += 1

            eval_loss /= eval_steps
            perplexity /= eval_steps

            if self.tb_writer:
                self.writer.add_scalar('Eval/Loss', eval_loss, eval_steps)
                self.writer.close()
                self.writer.add_scalar('Perplexity', perplexity, eval_steps)
                self.writer.close()
            logging.info(f'{datetime.now()} | Step: {step} | Eval Loss: {eval_loss} | Perplexity: {perplexity}')

        return None


class TokenizerTrainer:
    def __init__(self, ):
        pass

    def train(self, dataset_folder):
        dataset_files = []
        for _, _, filenames in os.walk(dataset_folder):
            for filename in filenames:
                if filename.endswith("token.dat"):
                    dataset_files.append(filename)

        dataset = load_dataset("text", data_dir=dataset_folder,
                               data_files=dataset_files, split='train')

        # split the dataset into training (80%) and testing (20%)
        d = dataset.train_test_split(test_size=0.2)
        with open('./models/tokenizer-train.txt', 'w') as file:
            for line in d['train']:
                file.write(line['text'] + '\n')
        with open('./models/tokenizer-test.txt', 'w') as file:
            for line in d['test']:
                file.write(line['text'] + '\n')

        special_tokens = ['[UNK]', '[PAD]', "[CLS]", "[SEP]", "[MASK]", '[LOG]', '[CALL]', '[ROOT]', '[START]',
                          '[STATE]', '[END]', '[OUTs]', '[INs]']

        # if you want to train the tokenizer on both sets
        files = ["./models/tokenizer-train.txt",
                 "./models/tokenizer-test.txt"]
        # training the tokenizer on the training set
        # files = ["train.txt"]
        # 30,522 vocab is BERT's default vocab size, feel free to tweak
        vocab_size = 2536
        # maximum sequence length, a lower setting will result in a faster training (when increasing batch size)
        max_length = 150
        # whether to truncate
        truncate_longer_samples = True

        # initialize the WordPiece tokenizer
        tokenizer = BertWordPieceTokenizer(unk_token="[UNK]")
        # train the tokenizer
        tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
        # enable truncation up to the maximum 512 tokens
        tokenizer.enable_truncation(max_length=max_length)
        model_path = "./models/tokenizer-model"
        # build the directory if not already there
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
            # save the tokenizer
            tokenizer.save_model(model_path)
            # dumping some of the tokenizer config to config file,
            # including special tokens, whether to lower case and the maximum sequence length
            with open(os.path.join(model_path, "config.json"), "w") as f:
                tokenizer_cfg = {
                    "do_lower_case": True,
                    "unk_token": "[UNK]",
                    "sep_token": "[SEP]",
                    "pad_token": "[PAD]",
                    "cls_token": "[CLS]",
                    "mask_token": "[MASK]",
                    "model_max_length": max_length,
                    "max_len": max_length,
                }
                json.dump(tokenizer_cfg, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-tokenizer",
                        required=False,
                        help="path to the dataset folder"
                        )

    parser.add_argument("--train-classifier",
                        required=False,
                        help="path to the dataset folder"
                        )

    args = parser.parse_args()
    if args.train_tokenizer:
        trainer = TokenizerTrainer()
        trainer.train(args.train_tokenizer)
    if args.train_classifier:
        dataset = TxtDataset(args.train_classifier)
        tokenizer = BertTokenizer(
            './models/tokenizer-model/vocab.txt')
        special_tokens = ['[log]', '[call]', '[root]', '[start]', '[state]', '[end]', '[outs]', '[ins]']
        tokenizer.add_tokens(special_tokens)

        model = ReformerLM(
            num_tokens=tokenizer.vocab_size,
            dim=512,
            depth=6,
            heads=8,
            max_seq_len=MAX_SEQ_LEN,
            causal=True
        )
        trainer = ReformerTrainer(dataset, model, tokenizer, train_batch_size=5, eval_batch_size=5)
        train_dataloader, eval_dataloader = trainer.build_dataloaders(train_test_split=0.20)
        model = trainer.train(epochs=3,
                              train_dataloader=train_dataloader,
                              eval_dataloader=eval_dataloader,
                              log_steps=10,
                              ckpt_steps=100,
                              ckpt_dir='./models/ckpts',
                              gradient_accumulation_steps=1)
        torch.save(model, './models/detect-model.bin')
