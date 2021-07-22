import config
import numpy as np
import pandas as pd
import dataset
import torch
from model import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.optim as optim
import engine
import time
import pickle

summary_context = pd.read_csv(config.TRAINING_FILE)
context = summary_context['text'].apply(dataset.replace_all).tolist()
summary = summary_context['headlines'].apply(dataset.replace_all).tolist()
train_context, valid_context, train_summary, valid_summary = train_test_split(context, summary, test_size=0.3, random_state=420)

# vocab = dataset.Vocab(train_context,
#                       config.spacy_en,
#                       preload=config.use_preload,
#                       file_name='train_context_vocab.pkl'
#                       )

vocab = dataset.Bert_vocab()

train_data = dataset.summary_dataset(vocab,
                                     train_context,
                                     train_summary
                                     )

valid_data = dataset.summary_dataset(vocab,
                                     valid_context,
                                     valid_summary
                                     )

train_iterator = DataLoader(train_data,
                            sampler = RandomSampler(train_data),
                            batch_size = config.BATCH_SIZE
                            )

valid_iterator = DataLoader(valid_data,
                            sampler = RandomSampler(valid_data),
                            batch_size = config.BATCH_SIZE
                            )

batch = next(iter(train_iterator))
pickle.dump(batch, open('small_batch_train.pkl', 'wb'))
batch = next(iter(valid_iterator))
pickle.dump(batch, open('small_batch_valid.pkl', 'wb'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(
    embed_size = config.embeded_size,
    vocab_size = vocab.tokenizer.get_vocab_size(),
    # vocab_size = config.MAX_VOCAB_SIZE,
    num_heads = config.num_heads,
    num_encoder_layers = config.num_encoder_layers,
    num_decoder_layers = config.num_decoder_layers,
    forward_expansion = config.forward_expansion,
    dropout = config.dropout,
    device = device,
    pad_idx = vocab.pad_ids,
    unk_idx = vocab.unk_ids
)

if config.use_pretrained:
    model.load_state_dict(torch.load(config.MODEL_PATH,
                                        map_location=torch.device(device)
                                        ))

model = model.to(device)
optimizer = AdamW(model.parameters(), lr = config.LR)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 5,
                                            num_training_steps = len(train_iterator) * config.N_EPOCHS
                                            )

n_epoch_stop = config.N_EPOCHS_STOP
min_loss = np.Inf

for epoch in range(config.N_EPOCHS):
    start_time = time.time()
    train_loss, train_score = engine.train(model,
                                           train_iterator,
                                           optimizer,
                                           device,
                                           scheduler,
                                           (epoch + 1, config.N_EPOCHS),
                                           vocab
                                           )
    torch.save(model.state_dict(), config.MODEL_PATH)
    valid_loss, valid_score = engine.evaluate(model,
                                              valid_iterator,
                                              device,
                                              vocab
                                              )
    end_time = time.time()
    epoch_mins, epoch_secs = engine.epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
          # f'| Train Bleu: {train_score * 100:.2f}%')
    print(f'\tValid Loss: {valid_loss:.3f}')
          # f'| Valid Bleu: {valid_score * 100:.2f}%')

    if valid_loss < min_loss:
        epochs_no_improve = 0
        min_val_loss = valid_loss
        torch.save(model.state_dict(), 'transformer_model_earlier_stop.model')
    else:
        epochs_no_improve += 1

    if epoch > 10 and epochs_no_improve == n_epoch_stop:
        print('Early Stopping')
        break
print('end')