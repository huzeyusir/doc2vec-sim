# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:20:48 2020

@author: huzeyu 
"""
import math
import pandas as pd
import jieba
from gensim import corpora, models
from jieba import analyse
import functools
import os
import pathlib
from sklearn.model_selection import train_test_split
import jieba.posseg as pseg
import codecs
import numpy
import gensim
import numpy as np
import sys
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import LabeledSentence
import logging 

a1 = 'D:\\bilibili\\TXT1.txt'
a2 = 'D:\\bilibili\\TXT2.txt'
def loadtxt(path):
    with open(path , encoding='utf_8_sig') as file:
        txt = file.read()  #去掉引号，以免造成行数减少quoting=3
    # print(str(txt))
    return str(txt)

def simlarityCalu(vector1,vector2):
    #计算余弦相似度
    vector1Mod=np.sqrt(vector1.dot(vector1))
    vector2Mod=np.sqrt(vector2.dot(vector2))
    if vector2Mod!=0 and vector1Mod!=0:
        simlarity=(vector1.dot(vector2))/(vector1Mod*vector2Mod)
    else:
        simlarity=0
    print("相似度：",simlarity)
    return simlarity

def keyword_extract(data):  
    # TF-IDF算法
    tfidf = analyse.extract_tags
    keywords = tfidf(data,15)
    return keywords

def word2vec(keywords,model): 
    wordvec_size = 192
    
    word_vec_all = numpy.zeros(wordvec_size)
    for data in keywords:  
        if model.wv.__contains__(data):
            word_vec_all= word_vec_all+ model.wv[data]
                
    return word_vec_all

def doc2vec(file_name):

    model = gensim.models.Doc2Vec.load('D:\\bilibili\\bilbili_corpus\\全部语料库.doc2vec')
    doc = [w for x in codecs.open(file_name,'r','utf-8').readlines() for w in jieba.cut(x.strip())]
    doc_vec_all = model.infer_vector(doc, alpha=0.01, steps=50)

    return doc_vec_all
def main():
    
    p1 = 'D:\\bilibili\\TXT1.txt'
    p2 = 'D:\\bilibili\\TXT2.txt'
    
    #word2vec方法
    # model = gensim.models.Word2Vec.load('D:\\bilibili\\bilbili_corpus\\全部语料库.word2vec')
    # a1 = loadtxt(p1)
    # a2 = loadtxt(p2)
    # k1 = keyword_extract(a1)
    # k2 = keyword_extract(a2)
    # vec1 = word2vec(k1,model)
    # vec2 = word2vec(k2,model)
    # simlarityCalu(vec1,vec2)
    
    vec1 = doc2vec(p1)
    vec2 = doc2vec(p2)
    # vec3 = doc2vec(p3)
    # 测试相似度
    simlarityCalu(vec1,vec2)

main()
