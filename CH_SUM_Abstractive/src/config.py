import os
import pandas as pd
import spacy
import tokenizers

BATCH_SIZE = 64
N_EPOCHS = 10
N_EPOCHS_STOP = 10
LR = 3e-5
embeded_size = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
forward_expansion = 4
dropout = 0.1
MODEL_PATH = 'transformer_model.model'
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    '../input/ch_bert_vocab2.txt',
    lowercase = True
)
TRAINING_FILE = '../input/train_tiny.csv'
# spacy_ch = spacy.load('zh_core_web_sm')
# MAX_VOCAB_SIZE = 18000
C_MAX_LEN = 150
S_MAX_LEN = 30
use_pretrained = False
# use_preload = False
