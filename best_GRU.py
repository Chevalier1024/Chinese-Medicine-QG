import logging
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from gensim.models import KeyedVectors
import torch
import random
from torch import nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import os
import re
from transformers import AdamW
from transformers.modeling_bert import BertModel, BertConfig
from QG_data_loader import preprocess_data, get_QG_loader, read_data
import json
import datetime
import pandas as pd
from rouge import Rouge

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'

start = datetime.datetime.now()

logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler('log.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

compute_rouge = Rouge()

# parameters
batch_size = 16
epochs = 50
early_stopping = 0
embed_dim = 200
enc_hid_dim = 1024
dec_hid_dim = 300
dropout = 0.2
clip = 1
learning_rate = 5e-5
model_name_or_path = '../user_data/pretrain_weight/chinese_roberta_wwm_large_ext_pytorch/'
beam_size = 5


def seed_everything(seed=2020):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()


def convert_ids_to_tokens(word_ids, oovs):
    words = []
    for word_id in word_ids:
        if word_id < vocab_size:
            words.append(id2word[word_id])
        else:
            pointer_idx = word_id - vocab_size
            words.append(oovs[pointer_idx])
    return words


def epoch_time(start_time, end_time):
    elapsed_secs = end_time - start_time
    elapsed_mins = elapsed_secs / 60
    return elapsed_mins, elapsed_secs


class Attention(nn.Module):
    """Implements additive attention and return the attention vector used to weight the values.
        Additive attention consists in concatenating key and query and then passing them trough a linear layer."""

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)

    def forward(self, hidden, enc_output, mask):
        # key = [batch size, dec hid dim]
        # queries = [batch size, src sent len, enc hid dim]

        batch_size = enc_output.shape[0]
        src_len = enc_output.shape[1]

        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1)

        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim]
        enc_output = self.attn(enc_output)
        # energy = [batch size, src sent len, dec hid dim]
        attention = torch.sum(hidden * enc_output, dim=2)
        attention.masked_fill_(mask, -math.inf)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.vocab_size = output_dim
        self.dec_hid_dim = dec_hid_dim

        self.attention = Attention(enc_hid_dim, dec_hid_dim)
        self.attention2 = Attention(enc_hid_dim, dec_hid_dim)

        self.rnn = nn.GRUCell(emb_dim+enc_hid_dim, dec_hid_dim)

        self.out = nn.Linear(enc_hid_dim + dec_hid_dim, output_dim)
        self.p = torch.nn.Linear(enc_hid_dim + dec_hid_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, embedded, enc_output, h1, pad_mask, max_n_oov, source_WORD_encoding_extended):
        # embedded = [batch_size, embed_dim]
        # queries = [batch_size, max_text_len, enc_hid_dim * 2]
        # key = [batch_size, dec_hid_dim]
        a = self.attention(h1, enc_output, pad_mask)
        weighted = torch.bmm(a.unsqueeze(1), enc_output).squeeze(1)

        rnn_input = torch.cat((embedded, weighted), dim=1)
        h1 = self.rnn(rnn_input, h1)

        a2 = self.attention2(h1, enc_output, pad_mask)
        weighted2 = torch.bmm(a2.unsqueeze(1), enc_output).squeeze(1)

        p_vocab = self.out(torch.cat((h1, weighted2), dim=1))
        p_vocab = F.softmax(p_vocab, dim=1)

        p_gen = self.sigmoid(self.p(torch.cat((h1, weighted2), dim=1))) # [bs, 1]

        ext_zeros = torch.zeros(p_vocab.size(0), max_n_oov).cuda()
        p_out = torch.cat([p_vocab, ext_zeros], dim=1)
        p_out = p_gen * p_out

        p_out.scatter_add_(1, source_WORD_encoding_extended, (1 - p_gen) * a2)

        return p_out, h1


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hid_dim, dec_hid_dim, dropout, embedding_matrix):
        super().__init__()

        self.vocab_size = vocab_size
        self.dec_hid_dim = dec_hid_dim
        self.dropout = nn.Dropout(dropout)
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.encoder = BertModel.from_pretrained(model_name_or_path)
        for param in self.encoder.parameters():
            param.requires_grad = True

        # self.transform = nn.Linear()

        self.decoder = Decoder(vocab_size, embed_dim, enc_hid_dim, dec_hid_dim)

    def forward(self, input_ids, segment_ids, input_mask,
                target_WORD_encoding, source_WORD_encoding_extended, max_n_oov, teacher_forcing_ratio=0.5):
        # text = [batch_size, max_text_len]
        # answer = [batch_size, max_answer_len]
        # question = [batch_size, max_question_len]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        pad_mask = (source_WORD_encoding_extended == word2id['<pad>'])

        target_WORD_encoding.masked_fill_(target_WORD_encoding >= self.vocab_size, word2id['<unk>'])
        question_embed = self.dropout(self.word_embedding(target_WORD_encoding))

        enc_outputs, _ = self.encoder(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        bs, max_q_len, _ = question_embed.size()

        # max_n_oov = source_WORD_encoding_extended.max().item() - self.vocab_size + 1
        # max_n_oov = max(max_n_oov, 1)

        outputs = torch.zeros(bs, max_q_len, self.vocab_size + max_n_oov).cuda()

        output = torch.LongTensor([word2id['<start>']] * bs).cuda()
        output = self.dropout(self.word_embedding(output))

        h1 = torch.zeros(bs, self.dec_hid_dim).cuda()

        for t in range(1, max_q_len):
            output, h1 = self.decoder(output, enc_outputs, h1, pad_mask, max_n_oov, source_WORD_encoding_extended)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            # print(teacher_force)
            top1 = output.max(1)[1]
            top1.masked_fill_(top1 >= self.vocab_size, word2id['<unk>'])
            top1 = self.dropout(self.word_embedding(top1))
            output = (question_embed[:, t] if teacher_force else top1)
        return (outputs+1e-12).log(), outputs


def train(model, dataloader, optimizer, clip):
    model.train()
    epoch_loss = 0
    start = time.time()
    for i, batch in enumerate(dataloader):
        source_WORD_encoding, source_len, \
        target_WORD_encoding, target_len, \
        source_WORD, target_WORD, \
        answer_position_encoding, answer_WORD, \
        source_WORD_encoding_extended, oovs, input_ids, segment_ids, input_mask, \
        origin_text, origin_answer, origin_question = [b.cuda() if isinstance(b, torch.Tensor) else b for b in batch]

        max_n_oov = source_WORD_encoding_extended.max().item() - vocab_size + 1
        max_n_oov = max(max_n_oov, 1)

        optimizer.zero_grad()
        log_p, _ = model(input_ids, segment_ids, input_mask,
                         target_WORD_encoding, source_WORD_encoding_extended, max_n_oov,
                         teacher_forcing_ratio=0.5)

        log_p = log_p[:, 1:]
        target_WORD_encoding = target_WORD_encoding[:, 1:]

        dec_pad_mask = (target_WORD_encoding == word2id['<pad>'])
        dec_valid_mask = ~dec_pad_mask
        dec_len = dec_valid_mask.float().sum(dim=1)

        nll_loss = -log_p.gather(2, target_WORD_encoding.unsqueeze(2)).squeeze(2)
        nll_loss.masked_fill_(dec_pad_mask, 0)
        nll_loss = nll_loss.sum(dim=1) / dec_len
        loss = nll_loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if i % int(len(dataloader) * 0.1) == int(len(dataloader) * 0.1) - 1:
            logger.info(
                f'Batch {i} Sentence loss {loss.item()} Time: {epoch_time(start, time.time())}')
            start = time.time()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def calculate_bleu(prediction, ground_truth, oovs):
    prediction[:, :, word2id['<unk>']] = -math.inf
    prediction[:, :, word2id['<pad>']] = -math.inf

    prediction = prediction.max(2)[1] # bs, len
    references = []
    hypotheses = []
    rouge_l = []
    for x, y, s in zip(ground_truth, prediction, oovs):
        y = convert_ids_to_tokens(y.tolist(), s)
        idx2 = y.index('<end>') if '<end>' in y else len(y)
        x = re.sub('\n', '', ' '.join(x))
        y = re.sub('\n', '', ' '.join(y[1:idx2]))
        references.append(x)
        hypotheses.append(y)
        score = compute_rouge.get_scores(y, x)
        rouge_l.append(score[0]['rouge-l']['f'])

    return sum(rouge_l)/len(rouge_l), references, hypotheses


def eval(model, dataloader):
    model.eval()

    epoch_loss = 0
    references = []
    hypotheses = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            source_WORD_encoding, source_len, \
            target_WORD_encoding, target_len, \
            source_WORD, target_WORD, \
            answer_position_encoding, answer_WORD, \
            source_WORD_encoding_extended, oovs, input_ids, segment_ids, input_mask, \
            origin_text, origin_answer, origin_question = [b.cuda() if isinstance(b, torch.Tensor) else b for b in batch]

            max_n_oov = source_WORD_encoding_extended.max().item() - vocab_size + 1
            max_n_oov = max(max_n_oov, 1)

            log_p, prediction = model(input_ids, segment_ids, input_mask,
                                      target_WORD_encoding, source_WORD_encoding_extended, max_n_oov,
                             teacher_forcing_ratio=0.)

            rouge_l, r, h = calculate_bleu(prediction, target_WORD, oovs)
            hypotheses.extend(h)
            references.extend(r)

            log_p = log_p[:, 1:]
            target_WORD_encoding = target_WORD_encoding[:, 1:]

            dec_pad_mask = (target_WORD_encoding == word2id['<pad>'])
            dec_valid_mask = ~dec_pad_mask
            dec_len = dec_valid_mask.float().sum(dim=1) - 1

            nll_loss = -log_p.gather(2, target_WORD_encoding.unsqueeze(2)).squeeze(2)
            nll_loss.masked_fill_(dec_pad_mask, 0)
            nll_loss = nll_loss.sum(dim=1) / dec_len
            loss = nll_loss.mean()

            epoch_loss += loss.item()
        metrics_dict = compute_rouge.get_scores(hypotheses, references)
        rouge_l = []
        for line in metrics_dict:
            rouge_l.append(line['rouge-l']['f'])

        logger.info(sum(rouge_l) / len(rouge_l))
        return epoch_loss / len(dataloader), sum(rouge_l) / len(rouge_l)

logger.info('read_data')
train_df = preprocess_data(split='train')
val_df = preprocess_data(split='dev')
test_df = preprocess_data(split='test')

train_loader = get_QG_loader(train_df, mode='train', batch_size=batch_size, shuffle=True, num_workers=1)
dev_loader = get_QG_loader(val_df, mode='dev', batch_size=batch_size, shuffle=False, num_workers=1)
test_loader = get_QG_loader(test_df, mode='test', batch_size=1, shuffle=False, num_workers=1)
bs_loader = get_QG_loader(val_df, mode='dev', batch_size=1, shuffle=False, num_workers=1)

logger.info('extract_vocab')
w2v_model = KeyedVectors.load_word2vec_format('../user_data/word2vec/tencent_char_embedding.bin')
word = w2v_model.vocab.keys()
embedding_matrix = w2v_model.vectors
word2id = {w: i for i, w in enumerate(word)}
id2word = {value: key for key, value in word2id.items()}

logger.info(list(word2id.items())[:10])
vocab_size = len(word2id)

logger.info('build model')

model = Seq2Seq(vocab_size, embed_dim, enc_hid_dim, dec_hid_dim, dropout, embedding_matrix)
# model = model.cuda()
model = nn.DataParallel(model, device_ids=[0, 1, 2]).cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if 'encoder' in n], 'lr': learning_rate},
        {'params': [p for n, p in param_optimizer if 'encoder' not in n]}
    ]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3, eps=1e-6)
best_rouge = 0.
logger.info('train and eval')
for epoch in range(epochs):
    start_time = time.time()

    logger.info(f'Epoch {epoch+1} training')
    train_loss = train(model, train_loader, optimizer, clip)
    logger.info(f'\nEpoch {epoch + 1} validation')
    valid_loss, rouge = eval(model, dev_loader)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if best_rouge < rouge:
        logger.info('save best weight')
        best_rouge = rouge
        torch.save(model.module.state_dict(), '../user_data/best_model/seq2seq_GRU.bin')
        early_stopping = 0
    else:
        early_stopping += 1

    logger.info(f'\nEpoch: {epoch + 1:02} completed | Time: {epoch_mins}m {epoch_secs}s')
    logger.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    logger.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | Val. rouge {rouge}'
                f'| BEST Val. rouge_l {best_rouge}\n\n')
    if early_stopping >= 5:
        break

# predict
logger.info('eval')
model.module.load_state_dict(torch.load('../user_data/best_model/seq2seq_GRU.bin'))
model.eval()

logger.info('greedy decoding')
hypotheses = []
references = []

with torch.no_grad():
    for i, batch in tqdm(enumerate(dev_loader)):
        source_WORD_encoding, source_len, \
        target_WORD_encoding, target_len, \
        source_WORD, target_WORD, \
        answer_position_encoding, answer_WORD, \
        source_WORD_encoding_extended, oovs, input_ids, segment_ids, input_mask, \
        origin_text, origin_answer, origin_question = [b.cuda() if isinstance(b, torch.Tensor) else b for b in batch]

        max_n_oov = source_WORD_encoding_extended.max().item() - vocab_size + 1
        max_n_oov = max(max_n_oov, 1)

        log_p, prediction = model(input_ids, segment_ids, input_mask,
                                  target_WORD_encoding, source_WORD_encoding_extended, max_n_oov,
                                  teacher_forcing_ratio=0.)

        bleu, r, h = calculate_bleu(prediction, target_WORD, oovs)
        hypotheses.extend(h)
        references.extend(r)

metrics_dict = compute_rouge.get_scores(hypotheses, references)
rouge_l = []
for line in metrics_dict:
    rouge_l.append(line['rouge-l']['f'])

logger.info(sum(rouge_l) / len(rouge_l))

logger.info('beam search decoding')
hypotheses = []
references = []

analysis_text = []
analysis_answer = []
analysis_question = []
analysis_key_context = []
pred_question = []
model = model.module
with torch.no_grad():
    for batch in tqdm(bs_loader):
        source_WORD_encoding, source_len, \
        target_WORD_encoding, target_len, \
        source_WORD, target_WORD, \
        answer_position_encoding, answer_WORD, \
        source_WORD_encoding_extended, oovs, input_ids, segment_ids, input_mask, \
        origin_text, origin_answer, origin_question = [b.cuda() if isinstance(b, torch.Tensor) else b for b in batch]

        k = beam_size
        references.append(' '.join(target_WORD[0]))
        analysis_text.append(origin_text[0])
        analysis_question.append(origin_question[0])
        analysis_answer.append(origin_answer[0])
        analysis_key_context.append(''.join(source_WORD[0]))

        pad_mask = (source_WORD_encoding_extended == word2id['<pad>'])

        enc_outputs, _ = model.encoder(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        enc_outputs = enc_outputs.repeat(k, 1, 1)
        source_WORD_encoding_extended = source_WORD_encoding_extended.repeat(k, 1)

        h1 = torch.zeros(k, dec_hid_dim).cuda()

        k_prev_words = torch.LongTensor([[word2id['<start>']]] * k).cuda()
        seqs = k_prev_words  # (k, 1)
        top_k_scores = torch.zeros(k, 1).cuda()  # (k, 1)
        complete_seqs = list()
        complete_seqs_scores = list()
        step = 1

        max_n_oov = source_WORD_encoding_extended.max().item() - vocab_size + 1
        max_n_oov = max(max_n_oov, 1)
        V = vocab_size + max_n_oov

        while True:
            k_prev_words.masked_fill_(k_prev_words >= vocab_size, word2id['<unk>'])
            embeddings = model.word_embedding(k_prev_words)
            embeddings = embeddings.squeeze(1)
            scores, h1 = model.decoder(embeddings, enc_outputs, h1, pad_mask, max_n_oov, source_WORD_encoding_extended)
            scores = (scores + 1e-12).log()
            scores[:, word2id['<unk>']] = -math.inf

            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // V  # (s)
            next_word_inds = top_k_words % V  # (s)
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word2id['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h1 = h1[prev_word_inds[incomplete_inds]]
            enc_outputs = enc_outputs[prev_word_inds[incomplete_inds]]
            source_WORD_encoding_extended = source_WORD_encoding_extended[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        if len(complete_seqs_scores) > 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            seq = seqs[0].tolist()
        hypothesis = []
        for w_id in seq:
            if w_id not in {word2id['<start>'], word2id['<end>'], word2id['<pad>']}:
                if w_id < vocab_size:
                    hypothesis.append(id2word[w_id])
                else:
                    pointer_idx = w_id - vocab_size
                    hypothesis.append(oovs[0][pointer_idx])
        hypotheses.append(' '.join(hypothesis))
        pred_question.append(''.join(hypothesis))

metrics_dict = compute_rouge.get_scores(hypotheses, references)
rouge_l = []
for line in metrics_dict:
    rouge_l.append(line['rouge-l']['f'])

logger.info(sum(rouge_l) / len(rouge_l))

analysis = pd.DataFrame()
analysis['text'] = analysis_text
analysis['answer'] = analysis_answer
analysis['key_context'] = analysis_key_context
analysis['question'] = analysis_question
analysis['pred_question'] = pred_question
analysis['text_len'] = list(map(len, analysis_text))
analysis['key_context_len'] = list(map(len, analysis_key_context))
analysis['answer_len'] = list(map(len, analysis_answer))

rouge_l = []
rouge = Rouge()
for true, pred in zip(analysis_question, pred_question):
    hypothesis = ' '.join(list(pred))
    reference = ' '.join(list(true))
    score = rouge.get_scores(hypothesis, reference)
    rouge_l.append(score[0]['rouge-l']['f'])
analysis['rouge_l'] = rouge_l
analysis.to_csv('analysis.csv', index=False, sep=',')


logger.info('predict')
test_question = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        source_WORD_encoding, source_len, \
        target_WORD_encoding, target_len, \
        source_WORD, target_WORD, \
        answer_position_encoding, answer_WORD, \
        source_WORD_encoding_extended, oovs, input_ids, segment_ids, input_mask, \
        origin_text, origin_answer, origin_question = [b.cuda() if isinstance(b, torch.Tensor) else b for b in batch]

        k = beam_size

        pad_mask = (source_WORD_encoding_extended == word2id['<pad>'])

        enc_outputs, _ = model.encoder(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        enc_outputs = enc_outputs.repeat(k, 1, 1)
        source_WORD_encoding_extended = source_WORD_encoding_extended.repeat(k, 1)

        h1 = torch.zeros(k, dec_hid_dim).cuda()

        k_prev_words = torch.LongTensor([[word2id['<start>']]] * k).cuda()
        seqs = k_prev_words  # (k, 1)
        top_k_scores = torch.zeros(k, 1).cuda()  # (k, 1)
        complete_seqs = list()
        complete_seqs_scores = list()
        step = 1

        max_n_oov = source_WORD_encoding_extended.max().item() - vocab_size + 1
        max_n_oov = max(max_n_oov, 1)
        V = vocab_size + max_n_oov

        while True:
            k_prev_words.masked_fill_(k_prev_words >= vocab_size, word2id['<unk>'])
            embeddings = model.word_embedding(k_prev_words)
            embeddings = embeddings.squeeze(1)
            scores, h1 = model.decoder(embeddings, enc_outputs, h1, pad_mask, max_n_oov, source_WORD_encoding_extended)
            scores = (scores + 1e-12).log()
            scores[:, word2id['<unk>']] = -math.inf

            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // V  # (s)
            next_word_inds = top_k_words % V  # (s)
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word2id['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h1 = h1[prev_word_inds[incomplete_inds]]
            enc_outputs = enc_outputs[prev_word_inds[incomplete_inds]]
            source_WORD_encoding_extended = source_WORD_encoding_extended[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        if len(complete_seqs_scores) > 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            seq = seqs[0].tolist()
        hypothesis = []
        for w_id in seq:
            if w_id not in {word2id['<start>'], word2id['<end>'], word2id['<pad>']}:
                if w_id < vocab_size:
                    hypothesis.append(id2word[w_id])
                else:
                    pointer_idx = w_id - vocab_size
                    hypothesis.append(oovs[0][pointer_idx])
        test_question.append(''.join(hypothesis))

with open('../data/juesai_1011.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
index = 0
submit = []
for line in data:
    new_line = {}
    new_annotations = []
    for annotation in line['annotations']:
        annotation['Q'] = test_question[index]
        index += 1
        new_annotations.append(annotation)
    new_line['id'] = line['id']
    new_line['text'] = line['text']
    new_line['annotations'] = new_annotations
    submit.append(new_line)

with open('../prediction_result/result_attention.json', 'w', encoding='utf-8') as f:
    json.dump(submit, f, indent=4, ensure_ascii=False)

end = datetime.datetime.now()
logger.info(end-start)
