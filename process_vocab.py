from collections import Counter
import itertools
import json
from gensim.models import word2vec, KeyedVectors
from tqdm import tqdm
import numpy as np
import re


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
            texts.append(list(clean_data(line['text'])))
            questions.append(list(annotation['Q']))
            answers.append(list(clean_data(annotation['A'])))
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
    word2idx["[CLS]"] = 4
    word2idx["[SEP]"] = 5
    idx = 6
    for word, value in counter.items():
        if value >= min_word_freq:
            word2idx[word] = idx
            idx += 1
    print('vocab size : {}'.format(len(word2idx)))
    return word2idx

print('read_data')
all_texts, all_questions, all_answers = read_data('../data/round1_train_0907.json')
test_1_texts, _, test_1_answers = read_data('../data/round1_test_0907.json')
test_2_texts, _, test_2_answers = read_data('../data/juesai_1011.json')

print('extract_vocab')
words = all_texts + all_questions + all_answers + test_1_texts + test_1_answers + test_2_texts + test_2_answers
vocab = extract_vocab(words, min_word_freq=5)

w2v_model = KeyedVectors.load_word2vec_format('../user_data/word2vec/Tencent_AILab_ChineseEmbedding.txt', binary=False)
print(len(w2v_model.vocab))

word2vec_dict = {}
embedding_max_value = 0
embedding_min_value = 1
count = 0
for word in tqdm(vocab.keys()):
    if word in w2v_model:
        word2vec_dict[word] = w2v_model[word]
        embedding_matrix = np.array(w2v_model[word])
        embedding_max_value = max(np.max(embedding_matrix), embedding_max_value)
        embedding_min_value = min(np.min(embedding_matrix), embedding_min_value)
        count += 1
    else:
        # word2vec_dict[word] = np.random.normal(0., 1.,  size=300)
        word2vec_dict[word] = np.zeros(300)

# word2vec_dict['<unk>'] = np.random.uniform(low=embedding_min_value, high=embedding_max_value, size=300)
# word2vec_dict['<end>'] = np.random.uniform(low=embedding_min_value, high=embedding_max_value, size=300)
# word2vec_dict['<start>'] = np.random.uniform(low=embedding_min_value, high=embedding_max_value, size=300)
# word2vec_dict['<pad>'] = np.random.uniform(low=embedding_min_value, high=embedding_max_value, size=300)

print(count)
print(len(word2vec_dict))

with open('../user_data/word2vec/char_embedding.bin', 'w') as f:
    f.write(str(len(word2vec_dict)-1) + ' 300\n')
    words = list(word2vec_dict.keys())
    for w in tqdm(words):
        if w != ' ':
            v = list(map(str, word2vec_dict[w]))
            f.write(w + ' ' + ' '.join(v) + '\n')