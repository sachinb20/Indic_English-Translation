# -*- coding: utf-8 -*-
"""Machine_Translation_Sachin_Working.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kZqQvGkpePCBozESWR89gQdQsflK3l-K
"""

import torch
import pandas as pd
import numpy as np
import os
import sys
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import math
from tempfile import TemporaryDirectory
from typing import Tuple
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, IterableDataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import Transformer
import math
from timeit import default_timer as timer

from preprocessing import load_data
from models import Seq2SeqTransformer
from MultiEncoder import make_model
from utils import  sequential_transforms, tensor_transform, create_mask, generate_square_subsequent_mask
from dataloader import tokenizer

# CUDA_VISIBLE_DEVICES=1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #set the device as GPU if available

EMB_SIZE = 400
NHEAD = 2
FFN_HID_DIM = 128
BATCH_SIZE = 16
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
lr = 0.0001
betas = (0.9, 0.98)
eps = 1e-9
NUM_EPOCHS = 36 #the result given will come if NUM_EPOCHS is set to 36.
split_size = 0.1

SRC_LANGUAGE = 'hi'
TGT_LANGUAGE = 'en'

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


#vocab_size = {}
#vocab_size[SRC_LANGUAGE] = 7000
#vocab_size[TGT_LANGUAGE] = 7000



Lang = 'All'
data, final_test_data = load_data(Lang)



answer_filename = 'answer_'+Lang +'.txt'
model_filename = 'model_'+Lang +'.pt'













#train test split using sklearn
train, eval = train_test_split(data, test_size = split_size, random_state = 42)
train = train.reset_index(drop = True)
eval = eval.reset_index(drop = True)
print(train.shape)
print(eval.shape)

token_transform,vocab_transform = tokenizer(SRC_LANGUAGE,TGT_LANGUAGE,train)

#defining the iterable class for creating the iterable dataset
#it takes two series as input namely hindi and english sentence series and
#generates a tuple of source and target sentence as follows
class MyIterableDataset(IterableDataset):
    def __init__(self, english_sentences, hindi_sentences):
        self.english_sentences = english_sentences
        self.hindi_sentences = hindi_sentences
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.english_sentences):
            raise StopIteration
        else:
            english_sentence = self.english_sentences[self.index]
            hindi_sentence = self.hindi_sentences[self.index]
            self.index += 1
            return hindi_sentence, english_sentence

# Example usage
train_iter = MyIterableDataset(train['english'], train['hindi'])
eval_iter = MyIterableDataset(eval['english'], eval['hindi'])







torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
print(SRC_VOCAB_SIZE)
print(TGT_VOCAB_SIZE)



#creating the model with the hyperparams specified as above
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

# initializes the weight matrices of the transformer model using Xavier uniform initialization
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE) #push the model to the device

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX) #cross entropy loss

optimizer = torch.optim.Adam(transformer.parameters(), lr=lr, betas=betas, eps=eps) #adam optimizer





text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor
    
'''creating the collate function to be passed in the dataloader which will basically apply
this function to every entry of the batch and make the data feedable to the model
'''

def collate_fn(batch):
    
    src_batch, tgt_batch = [], []
    
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch
    

train_iter = MyIterableDataset(train['english'], train['hindi'])
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
count = 1
for src, tgt in train_dataloader:
  if count == 5:
    break
  print(src.shape)
  count+=1

#checking the shapes of different batches

ntrain = train.shape[0]
neval = eval.shape[0]
print(ntrain)
print(neval)

#total number of samples in the train and eval dataset

ntrainbatches = int(np.ceil(ntrain/BATCH_SIZE))
nevalbatches = int(np.ceil(neval/BATCH_SIZE))
print("number of train batches ", ntrainbatches)
print("number of eval batches ", nevalbatches)

#total number of batches in the train and eval dataset

# Functions for training and evaluation on the whole dataset for one epoch

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = MyIterableDataset(train['english'], train['hindi'])
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    for src, tgt in train_dataloader:

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input,DEVICE)


        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    print("train losses ", losses)
    return losses / ntrainbatches


def evaluate(model):
    model.eval()
    losses = 0


    eval_iter = MyIterableDataset(eval['english'], eval['hindi'])
    val_dataloader = DataLoader(eval_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)


    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input,DEVICE)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    print("validation losses ", losses)
    return losses / nevalbatches

#!export CUDA_LAUNCH_BLOCKING=1 #might give error some time, just comment out if it does so

import gc
gc.collect()
torch.cuda.empty_cache()


#training loop
prev_val_loss = 0
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()

    train_loss = train_epoch(transformer, optimizer)

    end_time = timer()
    val_loss = evaluate(transformer)
    if epoch == 1:
        prev_val_loss = val_loss
    else:
        e = prev_val_loss - val_loss
        if e < 0.001:
            break
        prev_val_loss = val_loss

    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


torch.save(transformer.state_dict(),model_filename)
# #saving the model using pickle serialization
# import pickle
# with open('model_Hindi.pkl', 'wb') as file:
#     pickle.dump(transformer, file)
# print("File saved successfully")















































'''functions for decoding the final output tensor into the english sentence. We have used
two types of decoding techniques namely beam search decode and greedy decode. Although
we have used only the greedy decode scheme for our purpose for the reason that it takes
less '''

import heapq
import nltk
from nltk.translate.bleu_score import corpus_bleu


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0),DEVICE)
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size=3):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    # Initialize the beam with a single hypothesis
    beams = [(0.0, ys)]

    # Repeat beam expansion until max_len or EOS is reached
    for i in range(max_len-1):
        new_beams = []
        for score, ys in beams:
            # Check if the last token in the sequence is EOS
            if ys[-1] == EOS_IDX:
                new_beams.append((score, ys))
                continue

            memory = memory.to(DEVICE)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0),DEVICE)
                        .type(torch.bool)).to(DEVICE)
            tgt_mask = tgt_mask.unsqueeze(0)  # Add a new dimension at index 0
            tgt_mask = tgt_mask.repeat(2, tgt_mask.shape[1], tgt_mask.shape[1])
            print(ys.unsqueeze(0).shape, memory.squeeze(1).shape, tgt_mask.shape)
            out = model.decode(ys, memory, tgt_mask)
            out = out.squeeze(0)
            prob = model.generator(out[-1])
            top_probs, top_idxs = torch.topk(prob, beam_size)

            # Expand the beam with each possible next token
            for j in range(beam_size):
                next_word = top_idxs[j].item()
                score_j = score + top_probs[j].item()
                p = torch.tensor([next_word]).type_as(src.data)
                p = p.unsqueeze(0)
                print(ys.shape, p.shape)
                ys_j = torch.cat([ys, p], dim=0)
                new_beams.append((score_j, ys_j))

        # Keep only the top beam_size hypotheses
        beams = heapq.nlargest(beam_size, new_beams, key=lambda x: x[0])

    # Return the hypothesis with the highest score
    return beams[0][1]

# from pytorch_beam_search import seq2seq

# src and tgt language text transforms to convert raw strings into tensors indices


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.to(DEVICE)
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    src.to(DEVICE)
    src_mask.to(DEVICE)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

#example of translation
# print(translate(transformer, "ईमान लाओ और उसके रसूल के साथ होकर जिहाद करो"))
import pickle

#loading the model using pickle deserialization
# with open('results/model_Gujarati.pkl', 'rb') as file:
# loaded_model = pickle.load(file)
model=transformer
model.load_state_dict(torch.load(model_filename))
print("model loaded successfully")

#checking whether the loaded model is same as the transformer
print(final_test_data['sentence'][25], translate(model, final_test_data['sentence'][25]))

'''calculation of bleu score , uncomment the following code snippet and change the
hypothesis and reference sentence data in the corresponding fields to calculate the
corpus bleu score'''

# from torchtext.data.metrics import bleu_score
# actual = [token_transform['hi'](sentence) for sentence in eval['english'][:2000]]
# prediction = [token_transform['en'](translate(transformer, sentence)) for sentence in eval['hindi'][:2000]]
# Compute individual n-gram scores and their geometric mean
# BLEU-4

# print(f"BLEU score: {score: f}")

from tqdm import tqdm
#saving the predicted answers of the final_test_data
count = 0
with open(answer_filename, 'w', encoding = 'utf-8') as f:
  for sentence in tqdm(final_test_data['sentence']):
    translated = translate(transformer, sentence)
    # print(type(translated))
    count+=1
    f.write(translated + '\n')
