from tqdm import tqdm
import torch.nn as nn
import torch

def loss_fn(out, target, vocab):
    out = out.reshape(-1, out.shape[-1])
    target = target.reshape(-1)
    # the 1 here is from vocab pad index, if it is changed, plz do not forget to change here
    loss_F = nn.CrossEntropyLoss(ignore_index = vocab.pad_ids, reduction = 'sum')
    return loss_F(out, target)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, device, scheduler, epoch_tuple, vocab):
    epoch_loss = 0
    # epoch_bleu = 0
    total_len = 0
    model.train()
    TQDM = tqdm(enumerate(iterator), total=len(iterator), leave=False)
    for idx, batch in TQDM:
        for keys in batch:
            if keys in ['context_tokens', 'summary_tokens']:
                batch[keys] = batch[keys].to(device, dtype=torch.long)
            else:
                batch[keys] = batch[keys].to(device, dtype=torch.bool)
        context = batch['context_tokens']
        summary = batch['summary_tokens']
        context_mask = batch['context_mask']
        summary_mask = batch['summary_mask']
        optimizer.zero_grad()
        out = model(context,
                    summary[:, :-1],
                    context_mask,
                    summary_mask[:, :-1]
                    )
        # bleu_score = batch_predict_bleu_score(model, src, trg, trg_vocab, device)

        loss = loss_fn(out, summary[:, 1:], vocab)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        # epoch_bleu += bleu_score
        total_len += 1
        TQDM.set_description(f'Epoch [{epoch_tuple[0]}/{epoch_tuple[1]}]')
        TQDM.set_postfix({'loss': epoch_loss / total_len,
                          #'bleu': epoch_bleu / total_len,
                          # 'acc_pos': epoch_acc_pos / total_len
                          })
    return epoch_loss / total_len, None
        #, epoch_bleu / total_len


def evaluate(model, iterator, device, vocab):
    epoch_loss = 0
    # epoch_bleu = 0
    total_len = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            for keys in batch:
                if keys in ['context_tokens', 'summary_tokens']:
                    batch[keys] = batch[keys].to(device, dtype=torch.long)
                else:
                    batch[keys] = batch[keys].to(device, dtype=torch.bool)
            context = batch['context_tokens']
            summary = batch['summary_tokens']
            context_mask = batch['context_mask']
            summary_mask = batch['summary_mask']
            out = model(context,
                        summary[:, :-1],
                        context_mask,
                        summary_mask[:, :-1]
                        )
            # bleu_score = batch_predict_bleu_score(model, src, trg, trg_vocab, device)
            loss = loss_fn(out, summary[:, 1:], vocab)
            epoch_loss += loss.item()
            # epoch_bleu += bleu_score
            total_len += 1
    return epoch_loss / total_len, None
         # , epoch_bleu / total_len