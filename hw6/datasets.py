import os
import pickle
from multiprocessing import Pool
import numpy as np
import pandas as pd
import emoji
import torch
from torch.utils.data import Dataset
import jieba
from gensim.models import Word2Vec

class DcardDataset(Dataset):
    def __init__(self, x_path, y_path, dict_path, w2v_path, corpus_path=None, w2v_pretrain=True):
        self.x_path = x_path
        self.y_path = y_path
        self.dict_path = dict_path
        self.wv2_path = w2v_path
        self.corpus_path = corpus_path
        self.w2v_pretrain = w2v_pretrain
        
        self.preprocessing()

    def preprocessing(self):
        print("\n========= Start preprocessing ==========\n")
        jieba.load_userdict(self.dict_path)
        
        # Pre-train word embedding model
        if not self.w2v_pretrain:
            corpus = []

            for path in self.corpus_path:
                sentences = pd.read_csv(path).values[:, 1]

                P = Pool(4)
                tokens = P.map(self.tokenize, sentences)
                P.close()
                P.join()
                corpus.extend(tokens)

            w2v_model = Word2Vec(sentences=corpus, size=128, min_count=5, sg=1, iter=10)
            w2v_model.save(self.wv2_path)

        w2v_model = Word2Vec.load(self.wv2_path)
        
        sentences = pd.read_csv(self.x_path).values[:, 1]
        self.len = len(sentences)

        P = Pool(4)
        x_corpus = P.map(self.tokenize, sentences)
        P.close()
        P.join()

        
        # Transform the words into vectors with size = 250
        for i in range(self.len):
            if len(x_corpus[i]) > 100:
                x_corpus[i] = x_corpus[i][:100]

        self.x = np.zeros((self.len, 100, 128), dtype=np.float32)
        for i in range(self.len):
            for j in range(len(x_corpus[i])):
                if x_corpus[i][j] in w2v_model.wv.vocab:
                    self.x[i, j, :] = w2v_model.wv[x_corpus[i][j]]
        
        if self.y_path is not None:
            self.y = pd.read_csv(self.y_path).values[:, 1]
                
        del x_corpus
        print("\n========= Finish preprocessing ==========\n")


    def tokenize(self, sentence):
        words = jieba.lcut(emoji.demojize(sentence))
    
        return words

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.y_path is not None:
            return self.x[index], self.y[index]
        else:
            return self.x[index]

class BOW(Dataset):
    def __init__(self, x_path, y_path, dict_path, corpus_path=None, load=True):
        self.x_path = x_path
        self.y_path = y_path
        self.dict_path = dict_path
        self.corpus_path = corpus_path
        self.load = load

        self.preprocessing()

    def preprocessing(self):
        print("\n========= Start preprocessing ==========\n")
        jieba.load_userdict(self.dict_path)
        
        if not self.load:
            self.word2index = {}
            cnt = 0

            for path in self.corpus_path:
                sentences = pd.read_csv(path).values[:, 1]

                P = Pool(4)
                tokens = P.map(self.tokenize, sentences)
                P.close()
                P.join()
                
                for i in range(len(tokens)):
                    for token in tokens[i]:
                        if token not in self.word2index.keys():
                            self.word2index[token] = cnt
                            cnt += 1

            with open("./model/BOW+DNN/word2index.pkl", "wb") as f:
                pickle.dump(self.word2index, f, pickle.HIGHEST_PROTOCOL)            

        with open("./model/BOW+DNN/word2index.pkl", "rb") as f:
            self.word2index = pickle.load(f)
        
        sentences = pd.read_csv(self.x_path).values[:, 1]
        self.len = len(sentences)

        P = Pool(4)
        self.x = P.map(self.tokenize, sentences)
        P.close()
        P.join() 
        
        if self.y_path is not None:
            self.y = pd.read_csv(self.y_path).values[:, 1]
                
        print("\n========= Finish preprocessing ==========\n")


    def tokenize(self, sentence):
        words = jieba.lcut(emoji.demojize(sentence))
    
        return words

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        vector = np.zeros((len(self.word2index),), dtype=np.int32)
        
        for i in range(len(self.x[index])):
            vector[self.word2index[self.x[index][i]]] += 1
        
        if self.y_path is not None:
            return vector, self.y[index]
        else:
            return vector