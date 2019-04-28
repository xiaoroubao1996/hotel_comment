#!usr/bin/env python
#coding:utf-8

import jieba
import numpy as np
from gensim.models import Word2Vec 

class SentimentAnalysisWord2vec:
    #初始化
    def __init__(self,stopword):
        self.__readFile(stopword)

    #读取相关文库
    def __readFile(self,stopword):
        self.__stopword = []

        #停用词
        stopwordList = open(stopword,'r',encoding = 'utf-8')
        for s in stopwordList.readlines():
            try:
                s = s.replace('\r\n','').replace('\n','')
                self.__stopword.append(s) 
            except:
                print("stopword数据错误")
        stopwordList.close()

    def setSentences(self,sentences):
        self.__sentences = []
        num = 0
        for sentence in sentences:
            #去除左边空格
            try:
                sentenceWords = self._preDetail(sentence.lstrip())
                self.__sentences.append(sentenceWords)
                num = num + 1
            except:
                pass
    
    def setSentenceswithsplited(self,sentences):
        self.__sentences = sentences

    #结巴分词
    def _preDetail(self, sentence):
        wordsList = jieba.cut(sentence, cut_all=False)
        sentenceWords = []
        for w in wordsList:
            if w not in self.__stopword:
                sentenceWords.append(w)
        return sentenceWords

    def newModel(self, size = 100):
        # Create CBOW model
        self.__model = Word2Vec(self.__sentences , min_count = 5, size = size, window = 5) 
        self.__model.save(u'word2vec.model')
        return self.__model

    def getSentences(self):
        return self.__sentences

    def getModel(self):
        self.__model = Word2Vec.load('word2vec.model')
        return self.__model

    #将句子中的每一个词的vector合并并且取平均
    def sentencesVector(self, numFecture = 300):
        sentencesList = []
        for sentence in self.__sentences:
            sentenceList = np.zeros((numFecture,),dtype="float")
            numWords = 0
            for word in sentence:
                try:
                    sentenceList = np.add(sentenceList,self.__model[word])
                    numWords = numWords + 1
                except:
                    pass
            if numWords != 0:
                sentenceList = np.divide(sentenceList,numWords)
            sentencesList.append(np.array(sentenceList, dtype = 'float'))
        print("createListWithVector finished")
        return np.array(sentencesList)

def getSentimentAnalysisWord2vec():
    return SentimentAnalysisWord2vec('情感分析字典/停用词.txt')