import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from langconv import Converter #
import requests
import random
import hashlib
import time

def cat_to_chs(sentence): #传入参数为列表
    """
    将繁体转换成简体
    :param line:
    :return:
    """
    sentence = list(sentence)
    sentence = ",".join(sentence)
    sentence = Converter('zh-hans').convert(sentence)
    sentence.encode('utf-8')
    sentence = ''.join(sentence.split(","))
    return sentence


def baidu_translate(content, appid, secretKey, t_from='en', t_to='zh'):
    # print(content)
    if len(content) > 4891:
        return '输入请不要超过4891个字符！'
    salt = str(random.randint(0, 50))
    # 申请网站 http://api.fanyi.baidu.com/api/trans
    # 这里写你自己申请的
    appid = appid
    # 这里写你自己申请的
    secretKey = secretKey
    sign = appid + content + salt + secretKey
    sign = hashlib.md5(sign.encode(encoding='UTF-8')).hexdigest()
    head = {'q': f'{content}',
            'from': t_from,
            'to': t_to,
            'appid': f'{appid}',
            'salt': f'{salt}',
            'sign': f'{sign}'}
    j = requests.get('http://api.fanyi.baidu.com/api/trans/vip/translate', head)
    # print(j.json())
    t = j.json()
    res = j.json()['trans_result'][0]['dst']
    # print(res)
    return res

def translate(zh):
    time.sleep(1)
    en_s = baidu_translate(content=zh, appid='20201025000597954', secretKey='Sx9OcRVnaehySV3Ryg3g', t_from='zh', t_to='en')
    time.sleep(1)
    zh_s = baidu_translate(content=en_s, appid='20201025000597954', secretKey='Sx9OcRVnaehySV3Ryg3g', t_from='en', t_to='zh')

    if zh_s == zh:
        return None
    else:
        return zh_s

with open('../data/round1_train_0907.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
train, dev = train_test_split(data, shuffle=True, test_size=0.1, random_state=2020)

with open('../user_data/argument_data/CMRC/cmrc2018_train.json', 'r', encoding='utf-8') as f:
    argument_json = json.load(f)

new_train = []
count = 0
for paragraph in tqdm(train):
    line = {}
    line['id'] = paragraph['id']
    line['text'] = paragraph['text']
    annotations = []
    for qas in paragraph['annotations']:
        ann = {}
        Q = qas['Q']
        A = qas['A']
        ann['Q'] = Q
        ann['A'] = A
        annotations.append(ann)

        Q_ = translate(Q)
        A_ = translate(A)
        if Q_ is not None or A_ is not None:
            ann = {}
            ann['Q'] = Q_
            ann['A'] = A_
            annotations.append(ann)
            count += 1
print(count)

count = 0
for paragraphs in argument_json['data']:
    for paragraph in paragraphs['paragraphs']:
        line = {}
        line['id'] = paragraph['id']
        line['text'] = paragraph['context']
        annotations = []
        for qas in paragraph['qas']:
            ann = {}
            ann['Q'] = qas['question']
            ann['A'] = qas['answers'][0]['text']
            annotations.append(ann)
            count += 1
        line['annotations'] = annotations
        train.append(line)
print(count)

with open('../user_data/split_data/train.json', 'w', encoding='utf-8') as f:
    json.dump(train, f, indent=4, ensure_ascii=False)

with open('../user_data/split_data/dev.json', 'w', encoding='utf-8') as f:
    json.dump(dev, f, indent=4, ensure_ascii=False)


with open('../data/juesai_1011.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('../user_data/split_data/test.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)