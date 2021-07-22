import os
import pandas as pd
import spacy
import tokenizers

BATCH_SIZE = 64
N_EPOCHS = 10
N_EPOCHS_STOP = 10
LR = 3e-5
embeded_size = 256
num_heads = 8
num_encoder_layers = 4
num_decoder_layers = 4
forward_expansion = 4
dropout = 0.1
MODEL_PATH = 'transformer_model.model'
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    '../input/vocab_sos_eos.txt',
    lowercase = True
)
TRAINING_FILE = '../input/news_summary_more.csv'
spacy_en = spacy.load('en_core_web_sm')
# MAX_VOCAB_SIZE = 15000
C_MAX_LEN = 90
S_MAX_LEN = 30
use_pretrained = False
# use_preload = True
