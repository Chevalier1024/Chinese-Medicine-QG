import logging
import time
import json
from collections import Counter
import itertools
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import random
from torch import nn, optim
import torch.nn.functional as F
from nlgeval import NLGEval
import math
from tqdm import tqdm
import os
import re
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'


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


nlgeval = NLGEval()

# parameters
min_word_freq = 3
max_text_len = 500
max_question_len = 28  # <start> + 26 + <end>
max_answer_len = 150
batch_size = 128
epochs = 30
early_stopping = 0
embed_dim = 300
enc_hid_dim = 300
dec_hid_dim = 300
dropout = 0.2
clip = 1


def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = []
    questions = []
    answers = []
    for line in data:
        for annotation in line['annotations']:
            texts.append(list(line['text']))
            questions.append(list(annotation['Q']))
            answers.append(list(annotation['A']))
    return texts, questions, answers


def extract_vocab(iterable, min_word_freq=3):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    word2idx = dict()
    word2idx['<pad>'] = 0
    word2idx['<start>'] = 1
    word2idx['<end>'] = 2
    word2idx['<unk>'] = 3
    idx = 4
    for word, value in counter.items():
        if value >= min_word_freq:
            word2idx[word] = idx
            idx += 1
    logger.info('vocab size : {}'.format(len(word2idx)))
    return word2idx


def preprocess(data, vocab, max_len, if_go=True):
    processed_data = []
    length_data = []
    for line in data:
        encode = []
        if if_go:
            encode.append(vocab['<start>'])
        for word in line:
            encode.append(vocab[word] if word in vocab else vocab['<unk>'])
        if if_go:
            encode.append(vocab['<end>'])
        length_data.append(len(encode))
        encode = encode + [vocab['<pad>']] * (max_len - len(encode))
        processed_data.append(encode[:max_len])
    return processed_data, length_data


def convert_ids_to_tokens(line, id2word):
    word_data = [id2word[l] for l in line]
    return word_data


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

        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, key, queries):
        # key = [batch size, dec hid dim]
        # queries = [batch size, src sent len, enc hid dim]

        batch_size = queries.shape[0]
        src_len = queries.shape[1]

        # repeat encoder hidden state src_len times
        key = key.unsqueeze(1).repeat(1, src_len, 1)

        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim]
        energy = torch.tanh(self.attn(torch.cat((key, queries), dim=2)))
        # energy = [batch size, src sent len, dec hid dim]

        energy = energy.permute(0, 2, 1)
        # energy = [batch size, dec hid dim, src sent len]

        # v = [dec hid dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # v = [batch size, 1, dec hid dim]

        # This multiplication generate a number for each query
        attention = torch.bmm(v, energy).squeeze(1)
        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attention = Attention(enc_hid_dim, dec_hid_dim)
        self.rnn = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim, batch_first=True,)
        self.out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)


    def forward(self, embedded, queries, key):
        # embedded = [batch_size, embed_dim]
        # queries = [batch_size, max_text_len, enc_hid_dim * 2]
        # key = [batch_size, dec_hid_dim]
        self.rnn.flatten_parameters()

        a = self.attention(key, queries)

        # a = [batch size, src len]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]

        # queries = [batch size, src sent len, enc hid dim]

        weighted = torch.bmm(a, queries)
        # weighted = [batch size, 1, enc hid dim]
        # print(embedded.size())
        # print(weighted.size())
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch size, enc hid dim + emb dim]

        output, hidden = self.rnn(rnn_input, key.unsqueeze(0))

        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        output = self.out(torch.cat((output, weighted, embedded), dim=1))

        # output = [bsz, output dim]

        return output, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_encoder = nn.GRU(embed_dim, enc_hid_dim, bidirectional=True, batch_first=True)
        self.answer_encoder = nn.GRU(embed_dim, enc_hid_dim, bidirectional=True, batch_first=True)
        self.transform = nn.Linear(enc_hid_dim*2, dec_hid_dim)
        self.decoder = Decoder(vocab_size, embed_dim, enc_hid_dim*2, dec_hid_dim)

    def forward(self, text, answer, question, teacher_forcing_ratio=0.5):
        # text = [batch_size, max_text_len]
        # answer = [batch_size, max_answer_len]
        # question = [batch_size, max_question_len]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        self.text_encoder.flatten_parameters()
        self.answer_encoder.flatten_parameters()

        text_embed = self.dropout(self.embedding(text)) # [batch_size, max_text_len, embed_dim]
        answer_embed = self.dropout(self.embedding(answer)) # [batch_size, max_answer_len, embed_dim]
        question_embed = self.dropout(self.embedding(question)) # [batch_size, max_question_len, embed_dim]

        text_representation, _ = self.text_encoder(text_embed)
        answer_representation, _ = self.answer_encoder(answer_embed)

        batch_size = text_representation.shape[0]
        max_len = question_embed.shape[1]

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).cuda()

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer

        # first input to the decoder is the <start> tokens
        output = question_embed[:, 0].unsqueeze(1)

        hidden = self.transform(answer_representation[:, -1, :])

        for t in range(1, max_len):
            output, hidden = self.decoder(output, text_representation, hidden)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            # il primo 1 indica che il massimo viene cercato per ogni riga. Il secondo prende l'indice e non il valore
            top1 = output.max(1)[1]
            top1 = self.dropout(self.embedding(top1))
            top1 = top1.unsqueeze(1)
            output = (question_embed[:, t].unsqueeze(1) if teacher_force else top1)
        return outputs


def train(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    start = time.time()
    pw_criterion = nn.CrossEntropyLoss(ignore_index=0)
    for i, batch in enumerate(dataloader):
        batch = tuple(t.cuda() for t in batch)
        text, answer, question = batch

        optimizer.zero_grad()
        prediction = model(text, answer, question, 1.0)

        trg_sent_len = prediction.size(1)

        prediction = prediction[:, 1:].contiguous().view(-1, prediction.shape[-1])
        question = question[:, 1:].contiguous().view(-1)  # Find a way to avoid calling contiguous

        with torch.no_grad():
            pw_loss = pw_criterion(prediction,  question)

        loss = criterion(prediction,  question)

        # reshape to [trg sent len - 1, batch size]
        loss = loss.view(-1, trg_sent_len - 1)
        loss = loss.sum(1)
        loss = loss.mean(0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if i % int(len(dataloader) * 0.1) == int(len(dataloader) * 0.1) - 1:
            logger.info(
                f'Batch {i} Sentence loss {loss.item()} Time: {epoch_time(start, time.time())}')
            start = time.time()
        epoch_loss += pw_loss.item()

    return epoch_loss / len(dataloader)


def calculate_rouge(prediction, ground_truth, id2word):
    prediction = prediction.max(2)[1]
    references = []
    hypotheses = []
    for x, y in zip(ground_truth, prediction):
        x = convert_ids_to_tokens(x.tolist(), id2word)
        y = convert_ids_to_tokens(y.tolist(), id2word)
        idx1 = x.index('<end>') if '<end>' in x else len(x)
        idx2 = y.index('<end>') if '<end>' in y else len(y)
        x = re.sub('\n', '', ' '.join(x[1:idx1]))
        y = re.sub('\n', '', ' '.join(y[1:idx2]))
        references.append([x])
        hypotheses.append(y)

    metrics_dict = nlgeval.compute_metrics(references, hypotheses)
    return metrics_dict['ROUGE_L'], references, hypotheses


def eval(model, dataloader, criterion, id2word):
    model.eval()

    epoch_loss = 0
    references = []
    hypotheses = []
    pw_criterion = nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = tuple(t.cuda() for t in batch)
            text, answer, question = batch

            prediction = model(text, answer, question, 0)  # turn off teacher forcing

            sample_t = convert_ids_to_tokens(question[0].tolist(), id2word)
            sample_p = convert_ids_to_tokens(prediction[0].max(1)[1].tolist(), id2word)
            idx1 = sample_t.index('<end>') if '<end>' in sample_t else len(sample_t)
            idx2 = sample_p.index('<end>') if '<end>' in sample_p else len(sample_p)

            rouge_l, r, h = calculate_rouge(prediction, question, id2word)
            references.extend(r)
            hypotheses.extend(h)

            trg_sent_len = prediction.size(1)
            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            prediction = prediction[:, 1:].contiguous().view(-1, prediction.shape[-1])
            question = question[:, 1:].contiguous().view(-1)  # Find a way to avoid calling contiguous

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]
            with torch.no_grad():
                pw_loss = pw_criterion(prediction, question)

            loss = criterion(prediction, question)

            epoch_loss += pw_loss.item()

            loss = loss.view(-1, trg_sent_len - 1)
            loss = loss.sum(1)
            loss = loss.mean(0)

            if i % int(len(dataloader) * 0.1) == int(len(dataloader) * 0.1) - 1:
                logger.info(f'Batch {i} Sentence loss: {loss.item()} ROUGE_L score: {rouge_l}\n'
                         f'Target {sample_t[1:idx1]}\n'
                         f'Prediction {sample_p[1:idx2]}\n\n')

        metrics_dict = nlgeval.compute_metrics(references, hypotheses)
        logger.info(metrics_dict)
        return epoch_loss / len(dataloader), metrics_dict['ROUGE_L']

logger.info('read_data')
all_texts, all_questions, all_answers = read_data('./data/round1_train_0907.json')
test_texts, _, test_answers = read_data('./data/round1_test_0907.json')

logger.info('extract_vocab')
words = all_texts + all_questions + all_answers + test_texts + test_answers
vocab = extract_vocab(words, min_word_freq=min_word_freq)
vocab_size = len(vocab)
id2word = {value: key for key, value in vocab.items()}

logger.info('preprocess')
all_texts_id, _ = preprocess(all_texts, vocab, max_text_len, if_go=False)
all_questions_id, _ = preprocess(all_questions, vocab, max_question_len, if_go=True)
all_answers_id, _ = preprocess(all_answers, vocab, max_answer_len, if_go=False)

test_texts_id, _ = preprocess(test_texts, vocab, max_text_len, if_go=False)
test_answers_id, _ = preprocess(test_answers, vocab, max_answer_len, if_go=False)

train_id, valid_id = train_test_split(list(range(len(all_texts))), test_size=0.1, random_state=2020)

train_texts_id = np.array(all_texts_id)[train_id]
train_answers_id = np.array(all_answers_id)[train_id]
train_questions_id = np.array(all_questions_id)[train_id]

valid_texts_id = np.array(all_texts_id)[valid_id]
valid_answers_id = np.array(all_answers_id)[valid_id]
valid_questions_id = np.array(all_questions_id)[valid_id]

train_texts_id = torch.tensor(train_texts_id.tolist(), dtype=torch.long)
train_answers_id = torch.tensor(train_answers_id.tolist(), dtype=torch.long)
train_questions_id = torch.tensor(train_questions_id.tolist(), dtype=torch.long)

valid_texts_id = torch.tensor(valid_texts_id.tolist(), dtype=torch.long)
valid_answers_id = torch.tensor(valid_answers_id.tolist(), dtype=torch.long)
valid_questions_id = torch.tensor(valid_questions_id.tolist(), dtype=torch.long)

test_texts_id = torch.tensor(test_texts_id, dtype=torch.long)
test_answers_id = torch.tensor(test_answers_id, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(train_texts_id, train_answers_id, train_questions_id)
valid_dataset = torch.utils.data.TensorDataset(valid_texts_id, valid_answers_id, valid_questions_id)
test_dataset = torch.utils.data.TensorDataset(test_texts_id, test_answers_id)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = Seq2Seq(vocab_size, embed_dim, enc_hid_dim, dec_hid_dim, dropout)
model = nn.DataParallel(model, device_ids=[0, 1]).cuda()

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')  # Pad Index

best_rouge = 0.
logger.info('train and eval')
for epoch in range(epochs):
    start_time = time.time()

    logger.info(f'Epoch {epoch+1} training')
    train_loss = train(model, train_loader, optimizer, criterion, clip)
    logger.info(f'\nEpoch {epoch + 1} validation')
    valid_loss, rouge_l = eval(model, valid_loader, criterion, id2word)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if best_rouge < rouge_l:
        logger.info('save best weight')
        best_rouge = rouge_l
        torch.save(model.module.state_dict(), './model_save/best_weight.bin')
    else:
        early_stopping += 1
    if early_stopping >= 5:
        break
    logger.info(f'\nEpoch: {epoch + 1:02} completed | Time: {epoch_mins}m {epoch_secs}s')
    logger.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    logger.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | Val. ROUGE_L {rouge_l}'
                f'| BEST Val. ROUGE_L {best_rouge}\n\n')


# predict
logger.info('predict')
model.module.load_state_dict(torch.load('./model_save/best_weight.bin'))
model.eval()
test_question = []
with torch.no_grad():
    for i, batch in tqdm(enumerate(test_loader)):
        batch = tuple(t.cuda() for t in batch)
        text, answer = batch
        bs = text.size(0)
        q = torch.ones(bs, max_question_len).long().cuda()
        prediction = model(text, answer, q, 0)
        prediction = prediction.max(2)[1]
        for x in prediction:
            x = convert_ids_to_tokens(x.tolist(), id2word)
            idx1 = x.index('<end>') if '<end>' in x else len(x)
            test_question.append(''.join(x[1:idx1]))


with open('./data/round1_test_0907.json', 'r', encoding='utf-8') as f:
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

with open('./submit/submit.json', 'w', encoding='utf-8') as f:
    json.dump(submit, f, indent=4, ensure_ascii=False)
