import torch
import config
from torchtext.data.metrics import bleu_score
from model import *
import pickle
import dataset
import pandas as pd
from sklearn.model_selection import train_test_split


def greedy_search(single_summary_predict_out):
    return single_summary_predict_out[0][-1].argmax().item()

search_method = {'greedy' : greedy_search}

def summary_sentence(model, vocab, single_context, device, method = 'greedy'):
    single_context_padding_index = single_context == vocab.pad_ids
    single_context_padding_index = single_context_padding_index.to(device, dtype = torch.bool)
    single_summary_list = [vocab.sos_ids]
    model.eval()
    for i in range(config.S_MAX_LEN):
        single_summary = torch.LongTensor(single_summary_list).unsqueeze(0).to(device, dtype = torch.long)
        with torch.no_grad():
            single_summary_predict_out = model(single_context,
                                               single_summary,single_context_padding_index,
                                               None
                                               )
        best_guess = search_method[method](single_summary_predict_out)
        single_summary_list.append(best_guess)

        if single_summary_list[-1] == vocab.eos_ids:
            break
    return vocab.decode(single_summary_list)

def batch_predict(model, context, summary, vocab, device):
    targets = []
    outputs = []
    for idx, single_context in enumerate(context):
        single_context = single_context.unsqueeze(0)
        single_summary_pred = summary_sentence(model, vocab, single_context, device)
        single_summary_target = vocab.decode(summary[idx])
        outputs.append(single_summary_pred)
        targets.append(single_summary_target)
    return outputs, targets



if __name__ == '__main__':
    summary_context = pd.read_csv(config.TRAINING_FILE)
    context = summary_context['text'].apply(dataset.replace_all).tolist()
    summary = summary_context['headlines'].apply(dataset.replace_all).tolist()
    train_context, valid_context, train_summary, valid_summary = train_test_split(context, summary, test_size=0.3,
                                                                                  random_state=420)

    # vocab = dataset.Vocab(train_context,
    #                       config.spacy_en,
    #                       preload=config.use_preload,
    #                       file_name='train_context_vocab.pkl'
    #                       )
    vocab = dataset.Bert_vocab()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    batch = pickle.load(open('small_batch_valid.pkl', 'rb'))
    for keys in batch:
        if keys in ['context_tokens', 'summary_tokens']:
            batch[keys] = batch[keys].to(device, dtype=torch.long)
        else:
            batch[keys] = batch[keys].to(device, dtype=torch.bool)
        context = batch['context_tokens']
        summary = batch['summary_tokens']

    outputs, targets = batch_predict(model, context, summary, vocab, device)

    for i, j in zip(outputs, targets):
        print(i)
        print(j)
        print('###################')

    print('end')


