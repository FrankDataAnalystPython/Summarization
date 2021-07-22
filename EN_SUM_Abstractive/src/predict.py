import config
from dataset import *
from model import *
import engine
from utils import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import utils

text = "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human interventionThe $800 billion Paycheck Protection Program, which ended May 31, offered companies forgivable loans of up to $10 million to cover roughly two months of payroll and a handful of other expenses, such as rent. Applicants were not required to demonstrate any financial harm from the pandemic; they simply had to certify that “current economic uncertainty makes this loan request necessary” to support their continuing operations"

vocab = dataset.Bert_vocab()

input_data = dataset.summary_dataset(vocab, [text])
batch = input_data[0]


device = 'cpu'
model = Transformer(
    embed_size=config.embeded_size,
    vocab_size=vocab.tokenizer.get_vocab_size(),
    # vocab_size=config.MAX_VOCAB_SIZE,
    num_heads=config.num_heads,
    num_encoder_layers=config.num_encoder_layers,
    num_decoder_layers=config.num_decoder_layers,
    forward_expansion=config.forward_expansion,
    dropout=config.dropout,
    device=device,
    pad_idx=vocab.pad_ids,
    unk_idx=vocab.unk_ids
)

model.load_state_dict(torch.load(config.MODEL_PATH,
                                 map_location=torch.device(device)
                                 ))

pred_summary = utils.summary_sentence(model, vocab, batch['context_tokens'], device)

print(pred_summary)

print('end')