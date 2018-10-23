# -*- coding: UTF-8 -*-
import math
from heapq import nlargest
from itertools import product, count
import numpy as np
from gensim.models import word2vec
import os
import logging
import nltk
import multiprocessing
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
np.seterr(all='warn')

# output file paths
word2vec_embeddings_path = './word2vec/w2v_model'
training_path = './data/train_corpus.csv'
validate_path = './data/validate_corpus.csv'

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
max_textrank_iterations = 200
max_textrank_error = 0.005
unknown_word_during_textrank = []


def add_arguments(parser):
    parser.add_argument("--data_dir", type=str, default='/Users/wanfan01/Public', help="path of input data")
    parser.add_argument("--data_file_prefix", type=str, default='bytecup.corpus.train', help="prefix of file name")
    parser.add_argument("--text_rank_valid_data", action="store_true", default=False, help="input data type")


# ——————————————————data loading---------——————————————————————————————————————————
def load_data(file_dir, prefix, validate_data=False):
    contents_list = []
    titles_list = []
    if os.path.exists(file_dir):
        files = os.listdir(file_dir)
        for file in files:
            if file.startswith(prefix):
                file_path = os.path.join(file_dir, file)
                for line in open(file_path, 'r').readlines():
                    tmp = eval(line)
                    contents_list.append(tmp["content"])
                    if not validate_data:
                        titles_list.append(tmp["title"])
            else:
                continue
    return contents_list, titles_list


# ——————————————————data preprocessing and word2vec model training————————————————
def cut_sentences(content):
    """
    分句
    :param content:
    :return:list(sentence)
    """
    return tokenizer.tokenize(content)


def cut_word_stand(sentence):
    """
    词语全部转为小写、分词、去掉标点符号和去掉停用词
    :param sentence:
    :return:list(word)
    """
    stopwords = [word.strip('\n') for word in open('stopwords.txt').readlines()]
    cut_words = nltk.tokenize.WordPunctTokenizer().tokenize(sentence.lower())
    stemmer = nltk.PorterStemmer()
    # words = [stemmer.stem(word) for word in cut_words if word.isalnum() and word not in stopwords]
    words = [word for word in cut_words if word.isalnum() and word not in stopwords]
    return words


def content_process2wv(texts):
    """
    word2vec数据预处理，标准化等
    :param: texts
    :return: words
    """
    text_list = []
    for text in texts:
        text_list.append(cut_sentences(text))
    sentences = []
    for sentence_list in text_list:
        for line in sentence_list:
            sentences.append(cut_word_stand(line))
    return sentences


def get_word2vec_model(texts):
    """
    :param texts:
    :return:model
    """
    if not os.path.exists(word2vec_embeddings_path):
        sentences = content_process2wv(texts)
        print('make model-saved directory')
        os.makedirs(os.path.dirname(word2vec_embeddings_path))
        print('begin word2vec model training')
        model = word2vec.Word2Vec(sentences,
                                  size=300,
                                  window=10,
                                  min_count=5,
                                  workers=multiprocessing.cpu_count())
        model.save(word2vec_embeddings_path)
        print('end word2vec model training')
    else:
        print('loading trained word2vec model')
        model = word2vec.Word2Vec.load(word2vec_embeddings_path)

    return model


# ——————————————————construct sentence-undirected-weighted graph———————————————————
def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度
    :param vec1:
    :param vec2:
    :return:
    """
    tx = np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(sum(tx ** 2))
    cos22 = np.sqrt(sum(ty ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value


def compute_similarity_by_avg(model, sents_1, sents_2):
    """
    计算两个句子的相似度
    :param sents_1:
    :param sents_2:
    :return:
    """
    if len(sents_1) == 0 or len(sents_2) == 0:
        return 0.0
    vec1 = query_vector(sents_1[0], model)
    for word1 in sents_1[1:]:
        vec1 = vec1 + query_vector(word1, model)

    vec2 = query_vector(sents_2[0], model)
    for word2 in sents_2[1:]:
        vec2 = vec2 + query_vector(word2, model)

    similarity = cosine_similarity(vec1 / len(sents_1), vec2 / len(sents_2))
    return similarity


def query_vector(word, model, embedding_size=300):
    try:
        vec = model[word]
    except KeyError:
        unknown_word_during_textrank.append(word)
        vec = np.zeros(embedding_size, dtype=float)
    return vec


def create_graph(model, word_sent):
    """
    传入句子链表  返回句子之间相似度的图
    :param word_sent:
    :param model:
    :return:
    """
    num = len(word_sent)
    board = [[0.0 for _ in range(num)] for _ in range(num)]

    for i, j in product(range(num), repeat=2):
        if i != j:
            board[i][j] = compute_similarity_by_avg(model, word_sent[i], word_sent[j])
    return board


# ——————————————————text-rank algorithm iterations——————————————————————
def calculate_score(weight_graph, scores, i):
    """
    计算句子在图中的分数
    :param weight_graph:
    :param scores:
    :param i:
    :return:
    """
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0

    for j in range(length):
        denominator = 0.0
        # 计算分子
        fraction = weight_graph[j][i] * scores[j]
        # 计算分母
        for k in range(length):
            denominator += weight_graph[j][k]
        # the follow worthy discussing
        if denominator == 0:
            denominator = 1
        added_score += fraction / denominator
    # 算出最终的分数
    weighted_score = (1 - d) + d * added_score
    return weighted_score


def different(scores, old_scores):
    """
    判断前后分数有无变化
    :param scores:
    :param old_scores:
    :return:
    """
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= max_textrank_error:
            flag = True
            break
    return flag


def weight_sentences_rank(weight_graph):
    """
    输入相似度的图（矩阵)
    返回各个句子的分数
    :param weight_graph:
    :return:
    """
    # 初始分数设置为0.5
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]

    # begin iteration
    iter_step = 0
    while different(scores, old_scores) and iter_step < max_textrank_iterations:
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph, scores, i)
        iter_step += 1
    return scores


# ————————————————extract topN sentences from each content———————————————————————
def filter_words(model, sents):
    _sents = []
    for sentence in sents:
        for word in sentence:
            if word not in model:
                sentence.remove(word)
        if sentence:
            _sents.append(sentence)
    return _sents


def text_rank(model, content, n):
    tokens = cut_sentences(content)
    sentences = []
    sents_words = []
    for sentence in tokens:
        sentences.append(sentence)
        sents_words.append([word for word in cut_word_stand(sentence) if word])

    graph = create_graph(model, filter_words(model, sents_words))
    scores = weight_sentences_rank(graph)
    sent_selected = nlargest(n, zip(scores, count()))
    sent_index = []
    for i in range(n):
        sent_index.append(sent_selected[i][1])
    return [sentences[i] for i in sent_index]


def extract_top_n_sentences(model, n, contents, titles, validate_data=False):
    if not validate_data:
        out_file = open(training_path, 'w')
    else:
        out_file = open(validate_path, 'w')
    for index, content in enumerate(contents):
        print('text-rank extracting, content number: %d' % index)
        top_n_sentences = text_rank(model, content, n)
        simple_content = ''
        for sent in top_n_sentences:
            simple_content += sent
        line_dict = dict()
        line_dict['id'] = index
        line_dict['content'] = simple_content
        if not validate_data:
            line_dict['title'] = titles[index]
        line = str(line_dict) + '\n'
        out_file.write(line)
    out_file.flush()
    out_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    # loading initial training or validating data
    contents, titles = load_data(os.path.abspath(args.data_dir), args.data_file_prefix,
                                 validate_data=args.text_rank_valid_data)

    w2v_model = get_word2vec_model(contents)

    print(unknown_word_during_textrank)

    extract_top_n_sentences(w2v_model, 5, contents, titles, validate_data=args.text_rank_valid_data)
