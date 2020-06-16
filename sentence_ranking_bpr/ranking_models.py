import os
import re
from copy import deepcopy
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.metrics import label_ranking_loss
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from transformers import BertTokenizer,BertModel, BertConfig, AdamW

# import nltk
# ntlk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BPRLoss(nn.Module):
    def forward(self, prob_i, prob_j):
        return -(prob_i - prob_j).sigmoid().log().sum()


def encoude_input_sentences(input_sent, tokenizer):
    input_ids = []
    for sent in tqdm(input_sent, desc='Tokenizing'):
        tokenized_sent = word_tokenize(sent)[0:500]
        if len(tokenized_sent) > 0:
            encoded_sent = tokenizer.encode(tokenized_sent, add_special_tokens=True)
        else:
            encoded_sent = []
        input_ids.append(encoded_sent)
    return input_ids


def get_max_word_num(*sent_groups):
    return max([max([len(x) for x in sents]) for sents in sent_groups])


def pad_sequence(seq, maxnu):
    return np.array([np.pad(x, ((0, maxnu-len(x))), 'constant', constant_values=0) for x in seq])


def get_attention_masks(input_sent):
    return (input_sent > 0).astype(int)


class CallableTensorDataset(TensorDataset):
    def __call__(self, idx):
        return [X[idx] for X in self.tensors]


class PairwiseData:
    def __init__(self, tensor_dataset, probs, num_negative=10):
        self.probs = probs
        self.tensor_dataset = tensor_dataset
        self.num_samples = len(probs)
        self.num_negative = num_negative

    def negative_sampling(self, idx):
        neg_candidates = (self.probs < self.probs[idx]).nonzero().squeeze(1)
        num_candidates = neg_candidates.shape[0]
        negative_samples = neg_candidates[torch.randperm(num_candidates)[:min(num_candidates, self.num_negative)]]
        return negative_samples

    def __call__(self, sample_ids):
        if not isinstance(sample_ids, torch.LongTensor):
            sample_ids = torch.LongTensor([idx[0].item() for idx in sample_ids])

        negative_ids = torch.cat([self.negative_sampling(i) for i in sample_ids])
        positive_ids = sample_ids.unsqueeze(1).repeat(1, self.num_negative).reshape(-1)
        # return self.tensor_dataset(positive_ids), self.tensor_dataset(negative_ids)
        return positive_ids, negative_ids


def build_dataloader(train_val, test, batch_size, num_negative):
    train_inputs, train_masks, train_probs, val_inputs, val_masks, val_probs = map(torch.tensor, train_val)
    test_inputs, test_masks, test_probs = map(torch.tensor, test)

    train_collator = PairwiseData(CallableTensorDataset(train_inputs, train_masks), train_probs, num_negative)
    train_dataloader = DataLoader(TensorDataset(torch.arange(train_inputs.shape[0])), 
                                  batch_size=batch_size, 
                                  shuffle=False,
                                  collate_fn=train_collator)

    val_collator = PairwiseData(CallableTensorDataset(val_inputs, val_masks), val_probs, num_negative)
    val_dataloader = DataLoader(TensorDataset(torch.arange(val_inputs.shape[0])), 
                                batch_size=batch_size, 
                                shuffle=False,
                                collate_fn=val_collator)

    test_data = TensorDataset(test_inputs, test_masks, test_probs)
    test_dataloader = DataLoader(test_data, batch_size=2*batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, train_collator, val_collator


def bertrnn_process(train_sent, 
                    val_sent, 
                    test_sent, 
                    train_label, 
                    val_labels, 
                    test_labels,
                    batch_size,
                    num_negative,
                    base_bert_name='bert-base-uncased',
                    sci_bert_name='allenai/scibert_scivocab_uncased'):
    print(f'Process sentences with base BERT model: {base_bert_name}')
    # tokenize
    tokenizer1 = BertTokenizer.from_pretrained(base_bert_name)
    train_nor, val_nor, test_nor = map(lambda x: encoude_input_sentences(x, tokenizer1), [train_sent, val_sent, test_sent])

    # pad sequense
    maxnum = get_max_word_num(train_nor, val_nor, test_nor)
    train_nor, val_nor, test_nor = map(lambda x: pad_sequence(x, maxnum), [train_nor, val_nor, test_nor])

    # get attention
    train_nor_att, val_nor_att, test_nor_att = map(get_attention_masks, [train_nor, val_nor, test_nor])

    nor_train_val = (train_nor, train_nor_att, train_label, val_nor, val_nor_att, val_labels) 
    nor_test = (test_nor, test_nor_att, test_labels)
    *nor_loader, nor_train_collator, nor_val_collator = build_dataloader(nor_train_val, nor_test, batch_size, num_negative)

    print(f'Process sentences with SciBERT model: {sci_bert_name}')
    # tokenize, pad sequense, and get attention
    tokenizer2 = BertTokenizer.from_pretrained(sci_bert_name)
    train_sci, val_sci, test_sci = map(lambda x: encoude_input_sentences(x, tokenizer2), [train_sent, val_sent, test_sent])
    train_sci, val_sci, test_sci = map(lambda x: pad_sequence(x, maxnum), [train_sci, val_sci, test_sci])
    train_sci_att, val_sci_att, test_sci_att = map(get_attention_masks, [train_sci, val_sci, test_sci])
    
    sci_train_val = (train_sci, train_sci_att, train_label, val_sci, val_sci_att, val_labels)
    sci_test = (test_sci, test_sci_att, test_labels)
    *sci_loader, sci_train_collator, sci_val_collator = build_dataloader(sci_train_val, sci_test, batch_size, num_negative)
    return nor_loader, sci_loader, nor_train_collator, nor_val_collator, sci_train_collator, sci_val_collator


class BERTRNN(nn.Module):
    def __init__(self,
                 num_labels,
                 hidden_size=256,
                 dropout=0.2,
                 base_bert_name='bert-base-uncased',  # replace args['modelname1']
                 sci_bert_name='allenai/scibert_scivocab_uncased'):  # replace args['modelname1']
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.emb1 = BertModel.from_pretrained(base_bert_name, 
                                              num_labels=num_labels, 
                                              output_attentions=False,
                                              output_hidden_states=False)

        self.emb2 = BertModel.from_pretrained(sci_bert_name,
                                              num_labels=num_labels,
                                              output_attentions=False,
                                              output_hidden_states=False)

        self.emb1_size = self.emb1.config.hidden_size
        self.emb2_size = self.emb2.config.hidden_size
        self.emb_size = self.emb1_size + self.emb2_size

        self.lin1 = nn.Linear(self.emb_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_size, num_labels)

    def forward(self, data1, mask1, data2=None, mask2=None):
        emb1 = self.emb1(data1, attention_mask=mask1)
        pooler_output1 = emb1[1]
        emb2 = self.emb2(data2, attention_mask=mask2)
        pooler_output2 = emb2[1]
        pooler_output = self.dropout(torch.cat((pooler_output1, pooler_output2), 1))
        pooler_output = nn.functional.relu(self.lin1(pooler_output))
        out = self.lin2(pooler_output).sigmoid()
        return out.squeeze()


def train_(model, criteria, optimizer, dataloaders1, dataloaders2, collator1, collator2, epoch, device):
    model.train()
    epoch_loss = AverageMeter()
    pbar = tqdm(total=len(dataloaders1), desc='Training')
    for batch_idx, (pos_idx, neg_idx) in enumerate(dataloaders1):
        pos1 = collator1.tensor_dataset.tensors[0][pos_idx].to(device)
        pos_mask1 = collator1.tensor_dataset.tensors[1][pos_idx].to(device)
        neg1 = collator1.tensor_dataset.tensors[0][neg_idx].to(device)
        neg_mask1 = collator1.tensor_dataset.tensors[1][pos_idx].to(device)

        pos2 = collator2.tensor_dataset.tensors[0][pos_idx].to(device)
        pos_mask2 = collator2.tensor_dataset.tensors[1][pos_idx].to(device)
        neg2 = collator2.tensor_dataset.tensors[0][neg_idx].to(device)
        neg_mask2 = collator2.tensor_dataset.tensors[1][pos_idx].to(device)

        optimizer.zero_grad()
        out_pos = model.forward(pos1, pos_mask1, pos2, pos_mask2)
        out_neg = model.forward(neg1, neg_mask1, neg2, neg_mask2)

        loss = criteria(out_pos, out_neg)
        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.item(), n=pos1.shape[0])
        pbar.update()

    pbar.set_description(f'Epoch {epoch} done. Loss={epoch_loss.avg:.4e}')
    pbar.close()
    return epoch_loss


def validate_(model, criteria, optimizer, dataloaders1, dataloaders2, collator1, collator2, epoch, device):
    model.eval()
    epoch_loss = AverageMeter()
    pbar = tqdm(total=len(dataloaders1), desc='Validation')
    with torch.no_grad():
        for batch_idx, (pos_idx, neg_idx) in enumerate(dataloaders1):
            pos1 = collator1.tensor_dataset.tensors[0][pos_idx].to(device)
            pos_mask1 = collator1.tensor_dataset.tensors[1][pos_idx].to(device)
            neg1 = collator1.tensor_dataset.tensors[0][neg_idx].to(device)
            neg_mask1 = collator1.tensor_dataset.tensors[1][pos_idx].to(device)

            pos2 = collator2.tensor_dataset.tensors[0][pos_idx].to(device)
            pos_mask2 = collator2.tensor_dataset.tensors[1][pos_idx].to(device)
            neg2 = collator2.tensor_dataset.tensors[0][neg_idx].to(device)
            neg_mask2 = collator2.tensor_dataset.tensors[1][pos_idx].to(device)
            
            optimizer.zero_grad()
            out_pos = model.forward(pos1, pos_mask1, pos2, pos_mask2)
            out_neg = model.forward(neg1, neg_mask1, neg2, neg_mask2)

            loss = criteria(out_pos, out_neg)
            epoch_loss.update(loss.item(), n=pos1.shape[0])
            pbar.update()

        pbar.set_description(f'Epoch {epoch} done. Loss={epoch_loss.avg:.4e}')
        pbar.close()
    return epoch_loss


def test_(model, device, dataloaders1, dataloaders2):
    model.eval()
    preds = []
    pbar = tqdm(total=len(dataloaders1), desc='Testing')
    for batch_idx, (batch1, batch2) in tqdm(enumerate(zip(dataloaders1, dataloaders2))):
        data1, mask1, _ = map(lambda x: x.to(device), batch1)
        data2, mask2, _ = map(lambda x: x.to(device), batch2)
        with torch.no_grad(): 
            out = model(data1, mask1, data2, mask2)
            preds.append(out)
        pbar.update()
    pbar.close()
    preds = torch.cat(preds, dim=0)
    return preds.squeeze()


def test_and_save_results(model, device, test_nor_loader, test_sci_loader, test_df, out_path):
    preds = test_(model, device, test_nor_loader, test_sci_loader)
    test_df_with_pred = test_df.copy()
    test_df_with_pred['prob'] = preds.cpu().numpy()
    test_df_with_pred.to_csv(out_path)