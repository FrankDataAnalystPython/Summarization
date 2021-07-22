import pandas as pd
import torch
import config
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split

class Vocab:
    def __init__(self,
                 data,
                 spacy,
                 preload = False,
                 file_name = None
                 ):
        self.data = data
        self.preload = preload
        self.spacy = spacy
        self.special_tokens = ['<eos>', '<sos>', '<pad>', '<unk>']
        if self.preload:
            self.counter = pickle.load(open(file_name, 'rb'))
        else:
            self.counter = self.building_counter()
            pickle.dump(self.counter, open(file_name, 'wb'))
        self.builiding_vocab()
        self.pad_ids = self.stoi['<pad>']
        self.unk_ids = self.stoi['<unk>']
        self.sos_ids = self.stoi['<sos>']
        self.eos_ids = self.stoi['<eos>']

    def building_counter(self):
        counter = Counter()
        for sentences in self.data:
            tokens = [j.text.lower() for j in self.spacy(sentences)]
            counter.update(tokens)
        return counter

    def builiding_vocab(self):
        self.vocab = dict(self.counter.most_common(config.MAX_VOCAB_SIZE-4))
        self.itos = list(self.vocab)
        for i in self.special_tokens:
            self.itos.insert(0, i)
        for i in self.special_tokens:
            self.vocab[i] = 1
        self.stoi = {w : i for i, w in enumerate(self.itos)}

    def encode(self, sentences, max_len, add_special_tokens = True):
        tokens = [j.text.lower() for j in self.spacy(sentences)]
        if add_special_tokens:
            tokens = ['<sos>'] + tokens + ['<eos>']
        tokens_ids = [self.stoi[j] if j in self.stoi else self.stoi['<unk>'] for j in tokens]

        if len(tokens_ids) >= max_len:
            tokens_ids = tokens_ids[:max_len]
            tokens_ids[-1] = self.stoi['<eos>']
        else:
            while len(tokens_ids) < max_len:
                tokens_ids.append(self.stoi['<pad>'])

        return tokens_ids

    def decode(self, token_ids):
        tokens = [self.itos[j] for j in token_ids if self.itos[j] not in self.special_tokens[:-1]]
        tokens = ' '.join(tokens)
        return tokens

class Bert_vocab:
    def __init__(self):
        self.tokenizer = config.TOKENIZER
        self.sos_ids = self.tokenizer.get_vocab()['<sos>']
        self.eos_ids = self.tokenizer.get_vocab()['<eos>']
        self.pad_ids = self.tokenizer.get_vocab()['[PAD]']
        self.unk_ids = self.tokenizer.get_vocab()['[UNK]']
        # self.special_tokens = ['<eos>', '<sos>', '[PAD]', '[UNK]']
        # note the vocab.txt for bert has different special tokens, here I add the <eos> and <sos> into the end original vocab.txt

    def encode(self, sentences, max_len, add_special_tokens = True):
        tokens = self.tokenizer.encode(sentences, add_special_tokens = False)
        # Note the tokenizer here is bert, the add_special_tokens is True then [SEP][CLS] would be added, but we are using the sos and eos, so set it to False anyway
        if add_special_tokens:
            tokens_ids = [self.sos_ids] + tokens.ids + [self.eos_ids]

        if len(tokens_ids) >= max_len:
            tokens_ids = tokens_ids[:max_len]
            tokens_ids[-1] = self.eos_ids
        else:
            while len(tokens_ids) < max_len:
                tokens_ids.append(self.pad_ids)
        return tokens_ids

    def decode(self, token_ids):
        token_ids = list(token_ids)
        tokens = self.tokenizer.decode(token_ids).split(' ')
        tokens = [j for j in tokens if j not in ['<sos>', '<eos>']]
        tokens = ' '.join(tokens)
        return tokens

class summary_dataset:
    def __init__(self,
                 vocab,
                 context,
                 summary = None
                 ):
        self.context = context
        self.summary = summary
        self.vocab = vocab

    def __len__(self):
        return len(self.context)

    def __getitem__(self, item):
        context = self.context[item]
        context_tokens = self.vocab.encode(context, config.C_MAX_LEN)
        context_mask = [True if i == self.vocab.pad_ids else False for i in context_tokens]

        if self.summary is not None:
            summary = self.summary[item]
            summary_tokens = self.vocab.encode(summary, config.S_MAX_LEN)
            summary_mask = [True if i == self.vocab.pad_ids else False for i in summary_tokens]

        if self.summary is not None:
            return {'context_tokens' : torch.tensor(context_tokens, dtype = torch.long),
                    'context_mask' : torch.tensor(context_mask, dtype = torch.long),
                    'summary_tokens' : torch.tensor(summary_tokens, dtype = torch.long),
                    'summary_mask' : torch.tensor(summary_mask, dtype = torch.long)
                    }
        else:
            return {'context_tokens' : torch.tensor([context_tokens], dtype = torch.long),
                    'context_mask' : torch.tensor([context_mask], dtype = torch.long),
                    'summary_tokens' : None,
                    'summary_mask' : None
                    }

def replace_all(string):
    d = {'，' : ',',
         '”' : ',',
         '“' : '',
         '\n' : '',
         }
    for i in d:
        string = string.replace(i,d[i])
    return string

if __name__ == '__main__':
    summary_context = pd.read_csv(config.TRAINING_FILE)
    context = summary_context['text'].apply(replace_all).tolist()
    summary = summary_context['headlines'].apply(replace_all).tolist()
    train_context, valid_context, train_summary, valid_summary = train_test_split(context, summary, test_size = 0.3, random_state = 420)

    # vocab = Vocab(train_context,
    #               config.spacy_en,
    #               preload = config.use_preload,
    #               file_name = 'train_context_vocab.pkl'
    #               )

    vocab = Bert_vocab()
    summary_context_data = summary_dataset(vocab,
                                           train_context,
                                           train_summary)

    print('end')








