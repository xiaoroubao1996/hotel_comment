#!usr/bin/env python
#coding:utf-8

import jieba

class SentimentAnalysis:
    #初始化
    def __init__(self,sentiment,noword,adverb,stopword):
        self.__readFile(sentiment,noword,adverb,stopword)

    #读取相关文库
    def __readFile(self,sentiment,noword,adverb,stopword):
        self.__sentList = {}
        self.__noword = []
        self.__adverb = {}
        self.__stopword = []
        #情感词
        sentList = open(sentiment,'r',encoding = 'utf-8')
        for s in sentList.readlines():
            try:
                s = s.replace('\r\n','').replace('\n','')
                self.__sentList[s.split(' ')[0]] = s.split(' ')[1]
            except:
                print("sentList数据错误：")
        sentList.close()
        nowordList = open(noword,'r',encoding = 'utf-8')
        for s in nowordList.readlines():
            try:
                s = s.replace('\r\n','').replace('\n','')
                self.__noword.append(s)
            except:
                print("noword数据错误")
        nowordList.close()
        adverbList = open(adverb,'r',encoding = 'utf-8') 
        for s in adverbList.readlines():
            try:
                s = s.replace('\r\n','').replace('\n','')
                self.__adverb[s.split(',')[0]] = s.split(',')[1]
            except:
                print("adverb数据错误")
        adverbList.close()
        stopwordList = open(stopword,'r',encoding = 'utf-8')
        for s in stopwordList.readlines():
            try:
                s = s.replace('\r\n','').replace('\n','')
                self.__stopword.append(s) 
            except:
                print("stopword数据错误")
        stopwordList.close()

    def setSentence(self,sentence):
        self.__sentence = sentence.lstrip()
    #预处理
    def preDetail(self):
        wordsList = jieba.cut(self.__sentence, cut_all=False)
        newWords = {}
        i = 0
        for w in wordsList:
            if w not in self.__stopword:
                newWords[str(i)] =w
                i = i+1
        senWord = {}
        notWord = {}
        degreeWord = {}
        m = 0
        for index in newWords.keys():
            if newWords[index] in self.__sentList.keys() and newWords[index] not in self.__noword and newWords[index] not in self.__adverb.keys():
                #senWord[index] = self.__sentList[newWords[index].encode('utf-8')]
                senWord[index] = self.__sentList[newWords[index]]
            elif newWords[index] in self.__noword and newWords[index] not in self.__adverb.keys():
                notWord[index] = -1
            elif newWords[index] in self.__adverb.keys():
                degreeWord[index] = self.__adverb[newWords[index]]
            else:
                senWord[index] = 0
        return senWord,notWord,degreeWord,newWords

    def getScore(self):
        senWord,notWord,degreeWord,newWords = self.preDetail()
        W = 1
        score = 0
        # 存所有情感词的位置的列表
        senLoc = []
        notLoc = []
        degreeLoc = []
        for i in senWord.keys():
            senLoc.append(int(i))
        for i in notWord.keys():
            notLoc.append(int(i))
        for i in degreeWord.keys():
            degreeLoc.append(int(i))
        senLoc.sort()
        notLoc.sort()
        degreeLoc.sort()
        senloc = -1
        for i in range(0, len(newWords)):
            # 如果该词为情感词
            if i in senLoc:
                # loc为情感词位置列表的序号
                senloc += 1
                # 直接添加该情感词分数
                score += W * float(senWord[str(i)])
                # print "score = %f" % score
                if senloc < len(senLoc) - 1:
                    # 判断该情感词与下一情感词之间是否有否定词或程度副词
                    # j为绝对位置
                    if senLoc[senloc + 1] - senLoc[senloc] > 1:
                        for j in range(senLoc[senloc]+1, senLoc[senloc + 1]):
                            # 如果有否定词
                            if j in notLoc:
                                W *= -1
                            # 如果有程度副词
                            elif j in degreeLoc:
                                W *= float(degreeWord[str(j)])
                    else:
                        W = 1
            # i定位至下一个情感词
            if senloc < len(senLoc) - 1:
                i = senLoc[senloc + 1]

        return score
    def getSentence(self):
        return self.__sentence

def getAnalysis():
    return SentimentAnalysis('情感分析字典/情感词.txt', '情感分析字典/否定词.txt', '情感分析字典/程度副词.txt', '情感分析字典/停用词.txt')