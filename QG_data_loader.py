import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
import pickle
from gensim.models import word2vec, KeyedVectors
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import json
import re
from transformers.tokenization_bert import BertTokenizer

batch_size = 4
model_name_or_path = '../user_data/pretrain_weight/chinese_roberta_wwm_large_ext_pytorch/'
max_text_len = 153
max_question_len = 28  # <start> + 26 + <end>
max_answer_len = 100 # 150
max_len = max_text_len + max_answer_len + 3
# max_len = 512

w2v_model = KeyedVectors.load_word2vec_format('../user_data/word2vec/tencent_char_embedding.bin')
word = w2v_model.vocab.keys()
word2id = {w: i for i, w in enumerate(word)}
id2word = {value: key for key, value in word2id.items()}
print('word vocab size : {}'.format(len(word2id)))
print(list(word2id.items())[:10])

tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)


def clean_data(data):
    data = re.sub('\n', '', data)
    data = re.sub('\s+', '', data)
    data = re.sub('（\d+）', '', data)
    data = re.sub(r'\\n', '', data)
    data = re.sub('①|②|③|④|●|◆', '', data)
    return data


def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = []
    questions = []
    answers = []
    for line in data:
        for annotation in line['annotations']:
            texts.append(clean_data(line['text']))
            questions.append(annotation['Q'])
            answers.append(clean_data(annotation['A']))
    return texts, questions, answers


def preprocess_data(data_dir='../user_data/split_data/', split='train', n_process=1, PG=True):
    # word2id, id2word = vocab_read()
    path = data_dir + split + '.json'

    print(f'Preprocessing {split} dataset...')
    texts, questions, answers = read_data(path)
    PG = [PG] * len(texts)

    with Pool(n_process) as pool:
        data = list(tqdm(pool.imap(preprocess_single_example,
                                   zip(texts, questions, answers, PG)),
                         total=len(texts)))

    df = pd.DataFrame(data)
    print(f'Done! size: {len(df)}')
    return df


def extract_key_sentence(text, answer):
    if answer in text:
        start = text.index(answer)
        left = text[:start]
        right = text[start+len(answer):]
        left = re.split('(。|！|\!|\.|？|\?)', left)  # 保留分割符
        right = re.split('(。|！|\!|\.|？|\?)', right)  # 保留分割符
        key_sentence = left[-1] + answer + right[0]
    else:
        key_sentence = text
    return key_sentence


def get_answer_encoding(text, answer):
    start = None
    for index in range(len(text) - len(answer) + 1):
        if text[index:index+len(answer)] == answer:
            start = index
            break
    answer_encoding = [0] * len(text)
    if start is not None:
        answer_encoding[start:start+len(answer)] = [1] * len(answer)
    return answer_encoding


def preprocess_single_example(single_example):
    origin_text, origin_question, origin_answer, PG = single_example

    key_sentence = extract_key_sentence(origin_text, origin_answer)

    key_sentence = tokenizer.tokenize(key_sentence)
    answer = tokenizer.tokenize(origin_answer)

    # if len(key_sentence) + len(answer) + 3 >= 512:
    answer = answer[:max_answer_len]
    key_sentence = key_sentence[:max_len - len(answer) - 3]

    source_WORD = ["[CLS]"] + key_sentence + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(source_WORD)
    segment_ids = [0] * (len(key_sentence) + 2) + [1] * (len(answer) + 1)
    input_mask = [1] * len(input_ids)

    source_WORD_encoding = []
    source_WORD_encoding_extended = []
    oovs = []

    answer_position_encoding = [0] + get_answer_encoding(key_sentence, answer) + [0] + [1] * len(answer) + [0]

    for word in source_WORD:
        if word in word2id:
            source_WORD_encoding.append(word2id[word])
            source_WORD_encoding_extended.append(word2id[word])
        else:
            source_WORD_encoding.append(word2id['<unk>'])
            if word not in oovs:
                oovs.append(word)
            oov_num = oovs.index(word)
            source_WORD_encoding_extended.append(len(word2id) + oov_num)


    target_WORD = list(origin_question)
    target_WORD_encoding = []
    target_WORD_encoding.append(word2id['<start>'])
    for word in target_WORD:
        if word in word2id:
            target_WORD_encoding.append(word2id[word])
        # can be copied
        else:
            if not PG:
                if word in source_WORD:
                    target_WORD_encoding.append(len(word2id) + source_WORD.index(word))
                else:
                    target_WORD_encoding.append(word2id['<unk>'])
            else:
                if word in oovs:
                    target_WORD_encoding.append(len(word2id) + oovs.index(word))
                else:
                    target_WORD_encoding.append(word2id['<unk>'])
    target_WORD_encoding.append(word2id['<end>'])

    example = {
        'origin_text': origin_text,
        'origin_answer': origin_answer,
        'origin_question': origin_question,
        'answer_WORD': answer,  # 答案
        'source_WORD': source_WORD, # 上下文单词
        'source_WORD_encoding': source_WORD_encoding,  # 上下文单词编码为索引
        'source_WORD_encoding_extended': source_WORD_encoding_extended,  # 上下文单词扩展编码
        'source_len': len(source_WORD),  # 上下文长度
        'target_WORD': target_WORD, # 问题单词
        'target_WORD_encoding': target_WORD_encoding, # 问题编码为索引
        'target_len': len(target_WORD), # 问题长度
        'answer_position_encoding': answer_position_encoding,  # 答案编码为索引
        'input_ids': input_ids,
        'segment_ids': segment_ids,
        'input_mask': input_mask,
        'oovs': oovs, # oov词
    }
    return example


class SQuADDataset(Dataset):
    def __init__(self, df, split='train'):
        print('# Total size:', len(df))
        self.df = df
        if split != 'test':
            self.df = self.df.sort_values('source_len', ascending=False).reset_index()

        print(f'Done! Size: {len(self.df)}')

    def __getitem__(self, idx):
        return self.df.loc[idx]

    def __len__(self):
        return len(self.df)


def get_QG_loader(df, mode='train', **kwargs):
    dataset = SQuADDataset(df, mode)

    def qg_collate_fn(batch):
        batch = pd.DataFrame(batch).reset_index(drop=True)

        # Add <EOS> at the end of target target
        # batch.target_WORD_encoding = batch.target_WORD_encoding.apply(
        #     lambda x: x + [3])  # 3: word2id['<eos>']

        target_WORD_encoding = batch.target_WORD_encoding.apply(torch.LongTensor)
        target_WORD_encoding = pad_sequence(
            target_WORD_encoding, batch_first=True, padding_value=0)

        source_WORD_encoding = batch.source_WORD_encoding.apply(torch.LongTensor)
        source_WORD_encoding = pad_sequence(
            source_WORD_encoding, batch_first=True, padding_value=0)

        input_ids = batch.input_ids.apply(torch.LongTensor)
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=0)

        segment_ids = batch.segment_ids.apply(torch.LongTensor)
        segment_ids = pad_sequence(
            segment_ids, batch_first=True, padding_value=0)

        input_mask = batch.input_mask.apply(torch.LongTensor)
        input_mask = pad_sequence(
            input_mask, batch_first=True, padding_value=0)

        source_WORD_encoding_extended = batch.source_WORD_encoding_extended.apply(torch.LongTensor)
        source_WORD_encoding_extended = pad_sequence(
            source_WORD_encoding_extended, batch_first=True, padding_value=0)

        answer_position_encoding = batch.answer_position_encoding.apply(torch.LongTensor)
        answer_position_encoding = pad_sequence(
            answer_position_encoding, batch_first=True, padding_value=0)

        # Raw words
        source_WORD = batch.source_WORD.tolist()
        target_WORD = batch.target_WORD.tolist()
        answer_WORD = batch.answer_WORD.tolist()
        origin_text = batch.origin_text.tolist()
        origin_answer = batch.origin_answer.tolist()
        origin_question = batch.origin_question.tolist()

        source_len = batch.source_len.tolist()
        target_len = batch.target_len.tolist()

        oovs = batch.oovs.tolist()

        return source_WORD_encoding, source_len, \
               target_WORD_encoding, target_len, \
               source_WORD, target_WORD, \
               answer_position_encoding, answer_WORD, \
               source_WORD_encoding_extended, oovs, input_ids, segment_ids, input_mask, origin_text, origin_answer, origin_question

    return DataLoader(dataset, collate_fn=qg_collate_fn, **kwargs)

if __name__ == '__main__':

    val_df = preprocess_data(split='dev')
    print(val_df.columns)
