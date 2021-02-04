# 统计学搜索方法:tf-idf(词频,逆文本频率)的numpy实现
import numpy as np
from collections import Counter
import itertools
from visual import show_tfidf

docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]

docs_words = [d.replace(",", "").split(" ") for d in docs]
print(docs_words)
vocab = set(itertools.chain(*docs_words))                                           # 所有文本中出现的词汇
v2i = {v: i for i, v in enumerate(vocab)}                                           # 字典,词汇转下标
i2v = {i: v for v, i in v2i.items()}                                                # 下标转词汇


def safe_log(x):
    mask = x != 0                                                                   # 不为0的地方才能取log
    x[mask] = np.log(x[mask])
    return x

# 可以用字典做选择题!!!神妙的写法
tf_methods = {
        "log": lambda x: np.log(1+x),                                               # 对词汇出现频次取log
        "augmented": lambda x: 0.5 + 0.5 * x / np.max(x, axis=1, keepdims=True),
        "boolean": lambda x: np.minimum(x, 1),
        "log_avg": lambda x: (1 + safe_log(x)) / (1 + safe_log(np.mean(x, axis=1, keepdims=True))),
    }
idf_methods = {
        "log": lambda x: 1 + np.log(len(docs) / (x+1)),                             # 一个词出现的越多,idf越小,越没有区分力
        "prob": lambda x: np.maximum(0, np.log((len(docs) - x) / (x+1))),
        "len_norm": lambda x: x / (np.sum(np.square(x))+1),
    }


def get_tf(method="log"):
    _tf = np.zeros((len(vocab), len(docs)), dtype=np.float64)                       # [n_vocab, n_doc]
    for i, d in enumerate(docs_words):                                              # i代表第i个文本
        counter = Counter(d)
        for v in counter.keys():
            _tf[v2i[v], i] = counter[v] / counter.most_common(1)[0][1]              # (1):最常出现的那1个,返回形式如: [('am', 3)],[0][1]:取出现频次第一名的出现次数
            # 比如这个词是这个句子里出现最多的词,那它的tf = n/n = 1

    weighted_tf = tf_methods.get(method, None)
    if weighted_tf is None:
        raise ValueError
    return weighted_tf(_tf)                                                         # 用选定方法处理tf值


def get_idf(method="log"):
    df = np.zeros((len(i2v), 1))                                                    # 各个文本中某个单词出现的频次(1/0,即在某文本中出现或不出现)
    for i in range(len(i2v)):
        d_count = 0
        for d in docs_words:
            d_count += 1 if i2v[i] in d else 0                                      # 如果单词i在该文本中出现了df[i]+1
        df[i, 0] = d_count

    idf_fn = idf_methods.get(method, None)
    if idf_fn is None:
        raise ValueError
    return idf_fn(df)


def cosine_similarity(q, _tf_idf):                                                  # 化成两个单位向量,看其相似度
    unit_q = q / np.sqrt(np.sum(np.square(q), axis=0, keepdims=True))
    unit_ds = _tf_idf / np.sqrt(np.sum(np.square(_tf_idf), axis=0, keepdims=True))
    similarity = unit_ds.T.dot(unit_q).ravel()
    return similarity


def docs_score(q, len_norm=False):                                                  # q:搜索文本
    q_words = q.replace(",", "").split(" ")

    # 添加没出现过的词
    unknown_v = 0
    for v in set(q_words):
        if v not in v2i:
            v2i[v] = len(v2i)
            i2v[len(v2i)-1] = v
            unknown_v += 1
    if unknown_v > 0:
        _idf = np.concatenate((idf, np.zeros((unknown_v, 1), dtype=np.float64)), axis=0)
        _tf_idf = np.concatenate((tf_idf, np.zeros((unknown_v, tf_idf.shape[1]), dtype=np.float64)), axis=0)
    else:
        _idf, _tf_idf = idf, tf_idf
    counter = Counter(q_words)
    q_tf = np.zeros((len(_idf), 1), dtype=np.float64)                               # [n_vocab, 1]
    for v in counter.keys():
        q_tf[v2i[v], 0] = counter[v]

    q_vec = q_tf * _idf                                                             # q_tf和_idf都是[n_vocab, 1],这里的*是对位乘

    q_scores = cosine_similarity(q_vec, _tf_idf)                                    # 拿q和所有文本的tf-idf比,选择最接近的
    if len_norm:                                                                    # 对长度归一化
        len_docs = [len(d) for d in docs_words]
        q_scores = q_scores / np.array(len_docs)
    return q_scores


def get_keywords(n=2):                                                              # 测试前三篇文本的top2关键词
    for c in range(3):
        col = tf_idf[:, c]
        idx = np.argsort(col)[-n:]
        print("doc{}, top{} keywords {}".format(c, n, [i2v[i] for i in idx]))


tf = get_tf()           # [n_vocab, n_doc]
idf = get_idf()         # [n_vocab, 1]
tf_idf = tf * idf       # [n_vocab, n_doc]
print("tf shape(vecb in each docs): ", tf.shape)
print("\ntf samples:\n", tf[:2])                                                    # 看前两个词在15个文本中出现的频次
print("\nidf shape(vecb in all docs): ", idf.shape)
print("\nidf samples:\n", idf[:2])                                                  # 看前两个词在所有文本中的稀有度
print("\ntf_idf shape: ", tf_idf.shape)
print("\ntf_idf sample:\n", tf_idf[:2])


# test
get_keywords()
q = "I get a coffee cup"
scores = docs_score(q)
d_ids = scores.argsort()[-3:][::-1]                                                 # 选出相似度得分最大的三个idx,然后重新排序成由大到小
print("\ntop 3 docs for '{}':\n{}".format(q, [docs[i] for i in d_ids]))

show_tfidf(tf_idf.T, [i2v[i] for i in range(tf_idf.shape[0])], "tfidf_matrix")      # transpose一下,变成n_docs行,n_vocabs列