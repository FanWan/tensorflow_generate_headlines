import re
import collections
import pickle
import numpy as np
import html
import os
import nltk
from gensim.models import word2vec
from textrank_word2vec import tokenizer, word2vec_embeddings_path
from textrank_word2vec import training_path, validate_path

glove_embeddings_path = '/Users/wanfan01/Public/glove.840B.300d.txt'
embedding_matrix_save_path = './glove_embed/glove_embeddings.npy'


def get_init_data(train=False):
    """
        return init_data from input path
    """
    data_x = []
    data_y = []
    if train:
        for line in open(training_path, 'r').readlines():
            tmp = eval(line)
            data_x.append(preprocess(tmp["content"]))
            data_y.append(preprocess(tmp["title"]))
    else:
        for line in open(validate_path, 'r').readlines():
            tmp = eval(line)
            data_x.append(preprocess(tmp["content"]))
    return data_x, data_y


def clean_str(sent, keep_most=False):
    """
    Helper function to remove html, unneccessary spaces and punctuation.
    Args:
        sent: String.
        keep_most: Boolean. depending if True or False, we either
                   keep only letters and numbers or also other characters.
    Returns:
        processed sentence.
    """
    sent = sent.lower()
    sent = fix_up(sent)
    sent = re.sub(r"<br />", " ", sent)
    if keep_most:
        sent = re.sub(r"[^a-z0-9%!?.,:()/]", " ", sent)
    else:
        sent = re.sub(r"[^a-z0-9]", " ", sent)
    sent = re.sub(r"    ", " ", sent)
    sent = re.sub(r"   ", " ", sent)
    sent = re.sub(r"  ", " ", sent)
    sent = sent.strip()
    return sent


def fix_up(x):
    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def preprocess(text, keep_most=False):
    """
    Splits the text into sentences, preprocesses and tokenizes each sentence.
    Args:
        text: String. multiple sentences.
        keep_most: Boolean. depending if True or False, we either
                   keep only letters and numbers or also other characters.
    Returns:
        preprocessed and tokenized text.
    """
    tokenized = []
    for sentence in tokenizer.tokenize(text):
        sentence = clean_str(sentence)
        words = nltk.tokenize.WordPunctTokenizer().tokenize(sentence)
        for token in words:
            tokenized.append(token)
    return tokenized


def build_dict(train=False, word2index_path=None):
    article_list, title_list = get_init_data(train)
    if not os.path.exists(os.path.dirname(word2index_path)):
        # get word2index dictionary
        words = list()
        for sent in article_list + title_list:
            for word in sent:
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        # save word2index dictionary
        os.makedirs(os.path.dirname(word2index_path))
        with open(word2index_path, "wb") as f:
            pickle.dump(word_dict, f)
    else:
        with open(word2index_path, "rb") as f:
            word_dict = pickle.load(f)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))
    return word_dict, reversed_dict, article_list, title_list


def build_dataset(word_dict, article_list, article_max_len,
                  headline_max_len=None, headline_list=None, train=False):

    # converse word to index and make unknown word to the index of <unk>
    x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in article_list]

    # make the length of each sequence less than article_max_len
    x = [d[:article_max_len] for d in x]

    # padding each sentence if necessary
    x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x]
    
    if not train:
        return x
    else:
        y = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in headline_list]

        # make the length of each sequence less than headline_max_len - 1, cause that sequence y need be inserted a
        # begin tag <s> or an end tag </s> during training
        y = [d[:(headline_max_len - 1)] for d in y]
        return x, y


def batch_iter(inputs, outputs, batch_size, num_epochs):
    """
    creating batch-size data

    """
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def load_pretrained_embeddings(path):
    """
       loads pretrained embeddings. stores each embedding in a
       dictionary with its corresponding word
    """
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding_vector = np.array(values[1:], dtype='float32')
            embeddings[word] = embedding_vector
    return embeddings


def get_glove_embedding(word2ind, embedding_dim=300):
    """
       creates embedding matrix for each word in word2ind. if that words is in
       pretrained_embeddings, that vector is used. otherwise initialized
       randomly.
    """
    if os.path.exists(os.path.dirname(embedding_matrix_save_path)):
        return np.load(embedding_matrix_save_path)
    else:
        pretrained_embeddings = load_pretrained_embeddings(glove_embeddings_path)
        embedding_matrix = np.zeros((len(word2ind), embedding_dim), dtype=np.float32)
        for word, i in word2ind.items():
            if word in pretrained_embeddings.keys():
                embedding_matrix[i] = pretrained_embeddings[word]
            else:
                embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
                embedding_matrix[i] = embedding
        os.makedirs(os.path.dirname(embedding_matrix_save_path))
        np.save(embedding_matrix_save_path, embedding_matrix)
        return np.array(embedding_matrix)


def get_word2vec_embedding(word2ind, embedding_dim=300):
    model = word2vec.Word2Vec.load(word2vec_embeddings_path)
    del model
    word_vectors = model.wv
    embedding_matrix = np.zeros((len(word2ind), embedding_dim), dtype=np.float32)
    for word, i in word2ind.items():
        try:
            embedding_matrix[i] = word_vectors.word_vec(word)
        except KeyError:
            embedding_matrix[i] = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))

    return np.array(embedding_matrix)
