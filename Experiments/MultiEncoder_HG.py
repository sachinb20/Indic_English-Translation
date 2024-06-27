

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
from torch.nn import TransformerEncoder, TransformerEncoderLayer,TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import Dataset, IterableDataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import Transformer
import math
from timeit import default_timer as timer

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') #set the device as GPU if available

EMB_SIZE = 400
NHEAD = 2
FFN_HID_DIM = 512
BATCH_SIZE = 16
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
lr = 0.0001
betas = (0.9, 0.98)
eps = 1e-9
NUM_EPOCHS = 36 #the result given will come if NUM_EPOCHS is set to 36.
split_size = 0.1

SRC_LANGUAGE1 = 'Hindi'
SRC_LANGUAGE2 = 'Gujarati'
TGT_LANGUAGE = 'English'

# vocab_size = {}
# vocab_size[SRC_LANGUAGE] = 7000
# vocab_size[TGT_LANGUAGE] = 7000

#uncomment the following to mount while using colab
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

# !mkdir -p "/content/drive/My Drive/My Folder"

Lang = SRC_LANGUAGE1
answer_filename = 'answer_ME_HG.txt'
model_filename = 'model_ME_HG.pt'

#loading data from the desired directory
DATA_PATH = 'data/English_'+Lang+'.csv'
# TEST_PATH = '/kaggle/input/cs779-mt/eng_Hindi_data_dev_X.csv'
FINAL_TEST_DATA = 'data/English_'+Lang+'_val.csv'
data1 = pd.read_csv(DATA_PATH, header = None)
data1.columns = [SRC_LANGUAGE1, TGT_LANGUAGE]

# test = pd.read_csv(TEST_PATH, header = None)
# test.columns = ['sentence']
final_test_data1 = pd.read_csv(FINAL_TEST_DATA, header = None)
final_test_data1.columns = ['sentence']
# data.head()

def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

data1 = swap_columns(data1, SRC_LANGUAGE1, TGT_LANGUAGE)
print(data1.head())


data1[TGT_LANGUAGE] = data1[TGT_LANGUAGE].apply(str)
data1[SRC_LANGUAGE1] = data1[SRC_LANGUAGE1].apply(str)
final_test_data1['sentence'] = final_test_data1['sentence'].apply(str)

Lang = SRC_LANGUAGE2


#loading data from the desired directory
DATA_PATH = 'data/English_'+Lang+'.csv'
# TEST_PATH = '/kaggle/input/cs779-mt/eng_Hindi_data_dev_X.csv'
FINAL_TEST_DATA = 'data/English_'+Lang+'_val.csv'
data2 = pd.read_csv(DATA_PATH, header = None)
data2.columns = [SRC_LANGUAGE2, TGT_LANGUAGE]

# test = pd.read_csv(TEST_PATH, header = None)
# test.columns = ['sentence']
final_test_data2 = pd.read_csv(FINAL_TEST_DATA, header = None)
final_test_data2.columns = ['sentence']
# data.head()

def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

data2 = swap_columns(data2, SRC_LANGUAGE2, TGT_LANGUAGE)
print(data2.head())


data2[TGT_LANGUAGE] = data2[TGT_LANGUAGE].apply(str)
data2[SRC_LANGUAGE2] = data2[SRC_LANGUAGE2].apply(str)
final_test_data2['sentence'] = final_test_data2['sentence'].apply(str)

# !git clone "https://github.com/anoopkunchukuttan/indic_nlp_library"
# !git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
# !pip install Morfessor
# The path to the local git repo for Indic NLP library
INDIC_NLP_LIB_HOME=r"indic_nlp_library"

# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES="indic_nlp_resources"

import sys
sys.path.append(r'{}'.format(INDIC_NLP_LIB_HOME))
from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader
loader.load()

#train test split using sklearn
train1, eval1 = train_test_split(data1, test_size = split_size, random_state = 42)
train1 = train1.reset_index(drop = True)
eval1 = eval1.reset_index(drop = True)
print(train1.shape)
print(eval1.shape)
# train1.head()

#train test split using sklearn
train2, eval2 = train_test_split(data2, test_size = split_size, random_state = 42)
train2 = train2.reset_index(drop = True)
eval2 = eval2.reset_index(drop = True)
print(train2.shape)
print(eval2.shape)
# train2.head()

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
# train_iter = MyIterableDataset(train['english'], train['hindi'])
# eval_iter = MyIterableDataset(eval['english'], eval['hindi'])

import spacy
# !python -m spacy download en_core_web_sm
eng = spacy.load("en_core_web_sm")

from indicnlp.tokenize import indic_tokenize

def engTokenize(text):
    """
    Tokenize an English text and return a list of tokens
    """
    return [str(token.text) for token in eng.tokenizer(str(text))]

def hiTokenize(text):
    """
    Tokenize a German text and return a list of tokens
    """
    return [str(t) for t in indic_tokenize.trivial_tokenize(str(text))]

# def getTokens(data_iter, place):
#     """
#     Function to yield tokens from an iterator. Since, our iterator contains
#     tuple of sentences (source and target), `place` parameters defines for which
#     index to return the tokens for. `place=0` for source and `place=1` for target
#     """
#     for english, german in data_iter:
#         if place == 0:
#             yield engTokenize(english)
#         else:
#             yield hiTokenize(german)


########################################################################################################################################

class MyIterableEnglish(IterableDataset):
    def __init__(self, english_sentences):
        self.english_sentences = english_sentences
        # self.hindi_sentences = hindi_sentences
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.english_sentences):
            raise StopIteration
        else:
            english_sentence = self.english_sentences[self.index]
            # hindi_sentence = self.hindi_sentences[self.index]
            self.index += 1
            return english_sentence
        
train = pd.concat([train2[TGT_LANGUAGE], train1[TGT_LANGUAGE]])#, train3[TGT_LANGUAGE],train4[TGT_LANGUAGE],train5[TGT_LANGUAGE],train6[TGT_LANGUAGE],train7[TGT_LANGUAGE]], ignore_index=True)

vocab_size = {}
#vocab_size[SRC_LANGUAGE1] = 30000
vocab_size[TGT_LANGUAGE] = 40000

# Place-holders
token_transform = {}
vocab_transform = {}


token_transform[TGT_LANGUAGE] = engTokenize

# function to generate the tokens for each language
def yield_tokens(data_iter: Iterable) -> List[str]:
    language_index = {TGT_LANGUAGE: 0}

    for data_sample in data_iter:
        yield token_transform[TGT_LANGUAGE](data_sample)

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [TGT_LANGUAGE]:
    #create the iterator object of the dataset given
    train_iter = MyIterableEnglish(train)
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter),
                                                    min_freq=2,
                                                    specials=special_symbols,
                                                    special_first=True,
                                                    #max_tokens = vocab_size[ln]
                                                    )

#setting the default index to unknown index which means that it will assume the token to be unknown if
#it sees a word not in the dictionary.
for ln in [TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

print(len(vocab_transform[TGT_LANGUAGE]))
print('#################################################################################################################################################')

vocab_size1 = {}
vocab_size1[SRC_LANGUAGE1] = 30000
vocab_size1[TGT_LANGUAGE] = 30000

# Place-holders
token_transform1 = {}
vocab_transform1 = {}

token_transform1[SRC_LANGUAGE1] = hiTokenize
token_transform1[TGT_LANGUAGE] = engTokenize

# function to generate the tokens for each language
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE1: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform1[language](data_sample[language_index[language]])

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE1, TGT_LANGUAGE]:
    #create the iterator object of the dataset given
    train_iter = MyIterableDataset(train1[TGT_LANGUAGE], train1[SRC_LANGUAGE1])
    vocab_transform1[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=2,
                                                    specials=special_symbols,
                                                    special_first=True,
                                                    max_tokens = vocab_size1[ln]
                                                    )

#setting the default index to unknown index which means that it will assume the token to be unknown if
#it sees a word not in the dictionary.
for ln in [SRC_LANGUAGE1, TGT_LANGUAGE]:
  vocab_transform1[ln].set_default_index(UNK_IDX)

len(vocab_transform1[SRC_LANGUAGE1])

len(vocab_transform1[TGT_LANGUAGE])

print("tokenized hindi sentence ", token_transform1[TGT_LANGUAGE](train1[TGT_LANGUAGE][0]))
print("numericalized hindi sentence ", vocab_transform1[TGT_LANGUAGE](token_transform1[SRC_LANGUAGE1](train1[TGT_LANGUAGE][0])))
#check for the correct tokenization and numericalization

vocab_size2 = {}
vocab_size2[SRC_LANGUAGE2] = 30000
vocab_size2[TGT_LANGUAGE] = 30000

# Place-holders
token_transform2 = {}
vocab_transform2 = {}

token_transform2[SRC_LANGUAGE2] = hiTokenize
token_transform2[TGT_LANGUAGE] = engTokenize

# function to generate the tokens for each language
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE2: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform2[language](data_sample[language_index[language]])

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE2, TGT_LANGUAGE]:
    #create the iterator object of the dataset given
    train_iter = MyIterableDataset(train2[TGT_LANGUAGE], train2[SRC_LANGUAGE2])
    vocab_transform2[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=2,
                                                    specials=special_symbols,
                                                    special_first=True,
                                                    max_tokens = vocab_size2[ln]
                                                    )

#setting the default index to unknown index which means that it will assume the token to be unknown if
#it sees a word not in the dictionary.
for ln in [SRC_LANGUAGE2, TGT_LANGUAGE]:
  vocab_transform2[ln].set_default_index(UNK_IDX)

len(vocab_transform2[SRC_LANGUAGE2])

len(vocab_transform2[TGT_LANGUAGE])

print("tokenized hindi sentence ", token_transform2[TGT_LANGUAGE](train2[TGT_LANGUAGE][0]))
print("numericalized hindi sentence ", vocab_transform2[TGT_LANGUAGE](token_transform2[SRC_LANGUAGE2](train2[TGT_LANGUAGE][0])))
#check for the correct tokenization and numericalization



vocab_transform1[TGT_LANGUAGE] = vocab_transform[TGT_LANGUAGE] 
vocab_transform2[TGT_LANGUAGE] = vocab_transform[TGT_LANGUAGE] 


token_transform1[TGT_LANGUAGE] = token_transform[TGT_LANGUAGE] 
token_transform2[TGT_LANGUAGE] = token_transform[TGT_LANGUAGE] 



print('____________________________________________________________________________________________')
print(vocab_transform1[TGT_LANGUAGE](token_transform1[TGT_LANGUAGE]('Hello darkness my old friend')))
print(vocab_transform2[TGT_LANGUAGE](token_transform2[TGT_LANGUAGE]('Hello darkness my old friend')))
print('____________________________________________________________________________________________')



# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 7000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# The Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.encoder1 = TransformerEncoder(TransformerEncoderLayer(d_model=emb_size, nhead=nhead),num_layers = num_encoder_layers)
        self.encoder2 = TransformerEncoder(TransformerEncoderLayer(d_model=emb_size, nhead=nhead),num_layers = num_encoder_layers)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(d_model=emb_size, nhead=nhead),num_layers = num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor,
                Lang: str):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        if Lang == 'Hindi':
          mem = self.encoder1(src_emb, mask=src_mask, src_key_padding_mask = src_padding_mask)

        elif Lang == 'Gujarati':
          mem = self.encoder2(src_emb, mask=src_mask, src_key_padding_mask = src_padding_mask)

        outs = self.decoder(tgt_emb,mem, tgt_mask, None, tgt_key_padding_mask = tgt_padding_mask, memory_key_padding_mask = memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor, Lang: str):
        if Lang == 'Hindi':
          return self.encoder1(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

        elif Lang == 'Gujarati':
          return self.encoder2(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)


    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


#mask creation for preventing the knowledge of presence of elements in future time steps
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

#hyper-paramerter setting

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform1[SRC_LANGUAGE1])
TGT_VOCAB_SIZE = len(vocab_transform1[TGT_LANGUAGE])
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

'''creating the collate function to be passed in the dataloader which will basically apply
this function to every entry of the batch and make the data feedable to the model
'''
from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform1 = {}
for ln in [SRC_LANGUAGE1, TGT_LANGUAGE]:
    text_transform1[ln] = sequential_transforms(token_transform1[ln], #Tokenization
                                               vocab_transform1[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn1(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform1[SRC_LANGUAGE1](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform1[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform2 = {}
for ln in [SRC_LANGUAGE2, TGT_LANGUAGE]:
    text_transform2[ln] = sequential_transforms(token_transform2[ln], #Tokenization
                                               vocab_transform2[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn2(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform2[SRC_LANGUAGE2](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform2[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# train_iter = MyIterableDataset(train['english'], train['hindi'])
# train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
# count = 1
# for src, tgt in train_dataloader:
#   if count == 5:
#     break
#   print(src.shape)
#   count+=1

#checking the shapes of different batches



ntrain1 = train1.shape[0]
neval1 = eval1.shape[0]
print(ntrain1)
print(neval1)

#total number of samples in the train and eval dataset

ntrainbatches1 = int(np.ceil(ntrain1/BATCH_SIZE))
nevalbatches1 = int(np.ceil(neval1/BATCH_SIZE))
print("number of train1 batches ", ntrainbatches1)
print("number of eval1 batches ", nevalbatches1)

#total number of batches in the train and eval dataset

ntrain2 = train2.shape[0]
neval2 = eval2.shape[0]
print(ntrain2)
print(neval2)

#total number of samples in the train and eval dataset

ntrainbatches2 = int(np.ceil(ntrain2/BATCH_SIZE))
nevalbatches2 = int(np.ceil(neval2/BATCH_SIZE))
print("number of train batches ", ntrainbatches2)
print("number of eval batches ", nevalbatches2)

#total number of batches in the train and eval dataset

# Functions for training and evaluation on the whole dataset for one epoch

def train_epoch1(model, optimizer):
    model.train()
    losses = 0
    train_iter = MyIterableDataset(train1[TGT_LANGUAGE], train1[SRC_LANGUAGE1])
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn1)
    for src, tgt in train_dataloader:

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)


        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask,'Hindi')

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    print("train losses ", losses)
    return losses / ntrainbatches1

def train_epoch2(model, optimizer):
    model.train()
    losses = 0
    train_iter = MyIterableDataset(train2[TGT_LANGUAGE], train2[SRC_LANGUAGE2])
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn2)
    for src, tgt in train_dataloader:

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)


        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask,'Gujarati')

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    print("train losses ", losses)
    return losses / ntrainbatches2


def evaluate1(model):
    model.eval()
    losses = 0


    eval_iter = MyIterableDataset(eval1[TGT_LANGUAGE], eval1[SRC_LANGUAGE1])
    val_dataloader = DataLoader(eval_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn1)


    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask,'Hindi')

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    print("validation losses ", losses)
    return losses / nevalbatches1

def evaluate2(model):
    model.eval()
    losses = 0


    eval_iter = MyIterableDataset(eval2[TGT_LANGUAGE], eval2[SRC_LANGUAGE2])
    val_dataloader = DataLoader(eval_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn2)


    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask,'Gujarati')

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    print("validation losses ", losses)
    return losses / nevalbatches2

# !export CUDA_LAUNCH_BLOCKING=1 #might give error some time, just comment out if it does so

import gc
gc.collect()
torch.cuda.empty_cache()

# !nvidia-smi

#training loop
prev_val_loss = 0
for epoch in range(1, 2*NUM_EPOCHS+1):
    start_time = timer()

    train_loss1 = train_epoch1(transformer, optimizer)

    end_time = timer()
    val_loss1 = evaluate1(transformer)


    print((f"Epoch: {epoch}, Train loss: {train_loss1:.3f}, Val loss: {val_loss1:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    start_time = timer()

    train_loss2 = train_epoch2(transformer, optimizer)

    end_time = timer()
    val_loss2 = evaluate2(transformer)
    if epoch == 1:
        prev_val_loss1 = val_loss1
        prev_val_loss2 = val_loss2
    else:
        e1 = prev_val_loss1 - val_loss1
        e2 = prev_val_loss2 - val_loss2
        if e1 < 0.001 and e2 < 0.001:
            break
        prev_val_loss1 = val_loss1
        prev_val_loss2 = val_loss2
    print((f"Epoch: {epoch}, Train loss: {train_loss2:.3f}, Val loss: {val_loss2:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

torch.save(transformer.state_dict(),'MultiEncHG.pt')

'''functions for decoding the final output tensor into the english sentence. We have used
two types of decoding techniques namely beam search decode and greedy decode. Although
we have used only the greedy decode scheme for our purpose for the reason that it takes
less '''

# import heapq
# import nltk
# from nltk.translate.bleu_score import corpus_bleu


# def greedy_decode(model, src, src_mask, max_len, start_symbol):
#     src = src.to(DEVICE)
#     src_mask = src_mask.to(DEVICE)

#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
#     for i in range(max_len-1):
#         memory = memory.to(DEVICE)
#         tgt_mask = (generate_square_subsequent_mask(ys.size(0))
#                     .type(torch.bool)).to(DEVICE)
#         out = model.decode(ys, memory, tgt_mask)
#         out = out.transpose(0, 1)
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.item()

#         ys = torch.cat([ys,
#                         torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
#         if next_word == EOS_IDX:
#             break
#     return ys

# def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size=3):
#     src = src.to(DEVICE)
#     src_mask = src_mask.to(DEVICE)

#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

#     # Initialize the beam with a single hypothesis
#     beams = [(0.0, ys)]

#     # Repeat beam expansion until max_len or EOS is reached
#     for i in range(max_len-1):
#         new_beams = []
#         for score, ys in beams:
#             # Check if the last token in the sequence is EOS
#             if ys[-1] == EOS_IDX:
#                 new_beams.append((score, ys))
#                 continue

#             memory = memory.to(DEVICE)
#             tgt_mask = (generate_square_subsequent_mask(ys.size(0))
#                         .type(torch.bool)).to(DEVICE)
#             tgt_mask = tgt_mask.unsqueeze(0)  # Add a new dimension at index 0
#             tgt_mask = tgt_mask.repeat(2, tgt_mask.shape[1], tgt_mask.shape[1])
#             print(ys.unsqueeze(0).shape, memory.squeeze(1).shape, tgt_mask.shape)
#             out = model.decode(ys, memory, tgt_mask)
#             out = out.squeeze(0)
#             prob = model.generator(out[-1])
#             top_probs, top_idxs = torch.topk(prob, beam_size)

#             # Expand the beam with each possible next token
#             for j in range(beam_size):
#                 next_word = top_idxs[j].item()
#                 score_j = score + top_probs[j].item()
#                 p = torch.tensor([next_word]).type_as(src.data)
#                 p = p.unsqueeze(0)
#                 print(ys.shape, p.shape)
#                 ys_j = torch.cat([ys, p], dim=0)
#                 new_beams.append((score_j, ys_j))

#         # Keep only the top beam_size hypotheses
#         beams = heapq.nlargest(beam_size, new_beams, key=lambda x: x[0])

#     # Return the hypothesis with the highest score
#     return beams[0][1]


# # actual function to translate input sentence into target language
# def translate(model: torch.nn.Module, src_sentence: str):
#     model.eval()
#     src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
#     num_tokens = src.shape[0]
#     src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
#     tgt_tokens = greedy_decode(
#         model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
#     return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

# #example of translation
# # print(translate(transformer, "ईमान लाओ और उसके रसूल के साथ होकर जिहाद करो"))

# #checking whether the loaded model is same as the transformer
# print(final_test_data['sentence'][25], translate(loaded_model, final_test_data['sentence'][25]))

# '''calculation of bleu score , uncomment the following code snippet and change the
# hypothesis and reference sentence data in the corresponding fields to calculate the
# corpus bleu score'''

# # from torchtext.data.metrics import bleu_score
# # actual = [token_transform['hi'](sentence) for sentence in eval['english'][:2000]]
# # prediction = [token_transform['en'](translate(transformer, sentence)) for sentence in eval['hindi'][:2000]]
# # Compute individual n-gram scores and their geometric mean
# # BLEU-4

# # print(f"BLEU score: {score: f}")

# #saving the predicted answers of the final_test_data
# count = 0
# with open(answer_filename, 'w', encoding = 'utf-8') as f:
#   for sentence in final_test_data['sentence']:
#     translated = translate(transformer, sentence)
#     # print(type(translated))
#     count+=1
#     f.write(translated + '\n')

# print(len(final_test_data['sentence']))
# print(count)

# #checking the count

# c = 0
# with open(answer_filename, 'r', encoding = 'utf-8') as f:
#   for line in f:
#     c+=1
# print(c)

# #rechecking the count by loading the answer.txt
