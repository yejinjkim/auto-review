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
from torch.utils.data import TensorDataset, DataLoader
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

def build_dataloader(train_val, test, batch_size):
    train_inputs, train_masks, train_labels, val_inputs, val_masks, val_labels = map(torch.tensor, train_val)
    test_inputs, test_masks, test_labels = map(torch.tensor, test)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    validation_data = TensorDataset(val_inputs, val_masks, val_labels)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader


def bertrnn_process(train_sent, 
                    val_sent, 
                    test_sent, 
                    train_label, 
                    val_labels, 
                    test_labels,
                    batch_size,
                    base_bert_name='bert-base-uncased',
                    use_sci_bert=True, 
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
    nor_loader = build_dataloader(nor_train_val, nor_test, batch_size)
    
    if use_sci_bert:
        print(f'Process sentences with SciBERT model: {sci_bert_name}')
        # tokenize, pad sequense, and get attention
        tokenizer2 = BertTokenizer.from_pretrained(sci_bert_name)
        train_sci, val_sci, test_sci = map(lambda x: encoude_input_sentences(x, tokenizer2), [train_sent, val_sent, test_sent])
        train_sci, val_sci, test_sci = map(lambda x: pad_sequence(x, maxnum), [train_sci, val_sci, test_sci])
        train_sci_att, val_sci_att, test_sci_att = map(get_attention_masks, [train_sci, val_sci, test_sci])
        
        sci_train_val = (train_sci, train_sci_att, train_label, val_sci, val_sci_att, val_labels)
        sci_test = (test_sci, test_sci_att, test_labels)
        sci_loader = build_dataloader(sci_train_val, sci_test, batch_size)
    else:
        sci_loader = []
    return nor_loader, sci_loader


class BERTRNN(nn.Module):
    def __init__(self,
                 num_labels,
                 hidden_size=256,
                 dropout=0.2,
                 base_bert_name='bert-base-uncased',  # replace args['modelname1']
                 use_sci_bert=True, 
                 sci_bert_name='allenai/scibert_scivocab_uncased'):  # replace args['modelname1']
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_sci_bert = use_sci_bert
        
        self.emb1 = BertModel.from_pretrained(base_bert_name, 
                                              num_labels=num_labels, 
                                              output_attentions=False,
                                              output_hidden_states=False)
        self.emb1_size = self.emb1.config.hidden_size
        self.emb_size = self.emb1_size
        
        if use_sci_bert:
            self.emb2 = BertModel.from_pretrained(sci_bert_name,
                                                  num_labels=num_labels,
                                                  output_attentions=False,
                                                  output_hidden_states=False)
            self.emb2_size = self.emb2.config.hidden_size
            self.emb_size = self.emb1_size + self.emb2_size

        self.lin1 = nn.Linear(self.emb_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_size, num_labels)
        
    def forward(self, data1, mask1, data2=None, mask2=None):
        if self.use_sci_bert:
            emb1 = self.emb1(data1, attention_mask=mask1)
            pooler_output1 = emb1[1]
            emb2 = self.emb2(data2, attention_mask=mask2)
            pooler_output2 = emb2[1]
            pooler_output = self.dropout(torch.cat((pooler_output1, pooler_output2), 1))
            pooler_output = nn.functional.relu(self.lin1(pooler_output))
        else:
            emb1 =self.emb1(data1)
            pooler_output1 = self.dropout(emb1[1])
            pooler_output = nn.functional.relu(self.lin1(pooler_output1))

        out = self.lin2(pooler_output)
        return out


def train_(model, criteria, optimizer, dataloaders1, epoch, device, use_sci_bert, dataloaders2=None):
    model.train()
    epoch_loss = AverageMeter()
    pbar = tqdm(total=len(dataloaders1), desc='Training')
    if use_sci_bert:
        for batch_idx, batch in enumerate(zip(dataloaders1, dataloaders2)):
            data1, mask1, target1 = map(lambda x: x.to(device), batch[0])
            data2, mask2, target2 = map(lambda x: x.to(device), batch[1])
            
            optimizer.zero_grad()
            out = model.forward(data1, mask1, data2, mask2)
            loss = criteria(out, target1.float())
            loss.backward()
            optimizer.step()
        
            epoch_loss.update(loss.item(), n=data1.shape[0])  # TODO: check if batch goes in the first dimension.
            pbar.update()
    else:
        pbar = tqdm(total=len(dataloaders1), desc='Training')
        for batch_idx, batch in enumerate(dataloaders1):
            data1, mask1, target1 = map(lambda x: x.to(device), batch)
            optimizer.zero_grad()
            out = model.forward(data1, mask1)
            loss = criteria(out, target1.float())
            loss.backward()
            optimizer.step()
            epoch_loss.update(loss.item(), n=data1.shape[0])
            pbar.update()
    pbar.set_description(f'Epoch {epoch} done. Loss={epoch_loss.avg:.4e}')
    pbar.close()
    return epoch_loss


def validate_(model, dataloaders1, device, use_sci_bert, dataloaders2=None, pbar_msg='Validation/Testing'):
    model.eval()
    preds = []
    targets = []
    epoch_loss = AverageMeter()
    pbar = tqdm(total=len(dataloaders1), desc=pbar_msg)
    if use_sci_bert:
        for batch_idx, batch in tqdm(enumerate(zip(dataloaders1, dataloaders2))):
            data1, mask1, target1 = map(lambda x: x.to(device), batch[0])
            data2, mask2, target2 = map(lambda x: x.to(device), batch[1])
            with torch.no_grad(): 
                out = F.sigmoid(model(data1, mask1, data2, mask2))
                preds.append(out)
                targets.append(target1)
            pbar.update()
    else:
        for batch_idx, batch in enumerate(dataloaders1):
            data1, mask1, target1 = map(lambda x: x.to(device), batch)
            with torch.no_grad(): 
                out = F.sigmoid(model(data1, mask1, target1))
                preds.append(out)
                targets.append(target1)
            pbar.update()
    pbar.close()
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    return preds.squeeze(), targets.squeeze()


def test_and_save_results(model, test_nor_loader, device, use_sci_bert, test_sci_loader, test_df, out_path):
    outs, targets = validate_(model, test_nor_loader, device, use_sci_bert, test_sci_loader, pbar_msg='Generating labels for test set')
    test_df_with_pred = test_df.copy()
    test_df_with_pred['prob'] = outs.cpu().numpy()
    test_df_with_pred.to_csv(out_path)