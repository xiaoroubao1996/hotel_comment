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
                sentenceWords = self.preDetail(sentence.lstrip())
                self.__sentences.append(sentenceWords)
                num = num + 1
            except:
                pass
    
    def setSentencesExisted(self,sentences):
        self.__sentences = sentences

    #结巴分词
    def preDetail(self, sentence, stopword = False):
        wordsList = jieba.cut(sentence, cut_all=False)
        sentenceWords = []
        if stopword:
            for w in wordsList:
                if w not in self.__stopword:
                    sentenceWords.append(w)
        else:
            for w in wordsList:
                sentenceWords.append(w)
        return sentenceWords

    def newModel(self, size = 300, getFromFile = True):
        if getFromFile:
            #get the model which I have created
            self.__model = self.getModel()
        else:
            # Create CBOW model
            self.__model = Word2Vec(self.__sentences , min_count = 2, size = size, window = 5, negative = 15) 
            self.__model.save(u'word2vec.model')
        return self.__model

    def getSentences(self):
        return self.__sentences

    def getModel(self):
        self.__model = Word2Vec.load('word2vec.model')
        return self.__model

    #将句子中的每一个词的vector合并并且取平均
    def sentencesVector(self, numFecture = 300, test = None):
        if test == None:
            sentences = self.__sentences
        else:
            sentences = test

        sentencesList = []
        for sentence in sentences:
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
    return SentimentAnalysisWord2vec('corpus/停用词.txt')