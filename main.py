import pandas as pd
from sklearn.decomposition import PCA           #加载PCA算法包
import Sentiment_analysis as sa
import wordvec as wv
import savelord as sl
import SVMmodel as sm
path = './corpus/'
hotelComment = pd.read_csv(path + 'ChnSentiCorp_htl_all.csv')

#print('评论数目（总体）：%d' % hotelComment.shape[0])
#print('评论数目（正向）：%d' % hotelComment[hotelComment.label==1].shape[0])
#print('评论数目（负向）：%d' % hotelComment[hotelComment.label==0].shape[0])
#rows = [row for row in hotelComment]
#print(hotelComment)



###############################################################
#感情打分/不使用
###############################################################

#初始化分析
# s = sa.getAnalysis()
# for sentence in hotelComment["review"]:
#     s.setSentence(sentence)
#     print(s.getSentence())
#     print(s.getScore())

###############################################################


###############################################################
#机器学习/深度学习
###############################################################


#Word2vec
w = wv.getSentimentAnalysisWord2vec()

#get corpus from file
with open('./corpus/corpus.txt', 'r') as f:
    if len(f.read()) == 0:
        #Word2vec
        w.setSentences(hotelComment["review"])
        sentences = w.getSentences()
        sl.Write(sentences, './corpus/corpus.txt')
    else:
        sentences = sl.Read('./corpus/corpus.txt')
        w.setSentencesExisted(sentences)
print("sentences created")

#Create CBOW model
model = w.newModel(size = 300, getFromFile = True)


#print(len(model.wv.vectors))
#print(model.wv.similarity('酒店', '大堂'))
#print(model.wv.most_similar('偏僻',topn = 10))

#get all word with vector
with open('./corpus/sentencesWithVector.txt', 'r') as f:
    if len(f.read()) == 0:
        #set all word with vector
        sentencesWithVector = w.sentencesVector(numFecture = 300)
        sl.WriteInt(sentencesWithVector, './corpus/sentencesWithVector.txt')
    else:
        sentencesWithVector = sl.ReadInt('./corpus/sentencesWithVector.txt')
print("sentencesWithVector seted")


#print(sentencesWithVector.shape)

#PCA 300 -> 100
pca = PCA(n_components=100)
reducedModel=pca.fit_transform(sentencesWithVector)
print("PCA done")

# print(reducedModel.shape)
# print(len(reducedModel))
# print(len(hotelComment["label"]))


#SVM
SVM = sm.getSVMmodel(reducedModel, hotelComment["label"])

#get from file
#SVM.classication(getFromFile = True)

# print("SVM classication done")
# SVM.predictTestandTrain()
# print("Test : " + str(SVM.f1scoreTest()))
# print("Train : " + str(SVM.f1scoreTrain()))
# print("accuracy : " + str(SVM.accuracyTest()))

#cross validation
SVM.classicationCrossValisdation()
print("cross validation done")


#随机输入测试
testSentence = "这家酒店的环境十分糟糕，以后再也不来了"
testSentenceWords = w.preDetail(testSentence)
testSentencesWords = []
testSentencesWords.append(testSentenceWords)
testSentencesWithVector = w.sentencesVector(test = testSentencesWords)
testReducedModel = pca.transform(testSentencesWithVector)
print(testSentence + "result : "+ str(SVM.predictNewSentence(testReducedModel)))
