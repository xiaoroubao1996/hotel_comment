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
#感情打分
##################################################################

#初始化分析
# s = sa.getAnalysis()
# for sentence in hotelComment["review"]:
#     s.setSentence(sentence)
#     print(s.getSentence())
#     print(s.getScore())

################################################################

#Word2vec
w = wv.getSentimentAnalysisWord2vec()

#get corpus by file
with open('./corpus/corpus.txt', 'r') as f:
    if len(f.read()) == 0:
        #Word2vec
        w.setSentences(hotelComment["review"])
        sentences = w.getSentences()
        sl.Write(sentences)
    else:
        sentences = sl.Read()
        w.setSentenceswithsplited(sentences)
print("sentences created")

#Create CBOW model
#model = w.newModel(size = 300)

#get the model which I have created
model = w.getModel()


#print(len(model.wv.vectors))
#print(model.wv.similarity('服务员', '皇冠'))

#set all word with vector
sentencesWithVector = w.sentencesVector(numFecture = 300)

#print(sentencesWithVector.shape)

#PCA 300 -> 100
pca = PCA(n_components=100)
reduced_model=pca.fit_transform(sentencesWithVector)

print(reduced_model.shape)


print(len(reduced_model))
print(len(hotelComment["label"]))
#SVM
SVM = sm.getSVMmodel(reduced_model, hotelComment["label"])
SVM.classication()
SVM.predict()
print(SVM.f1score())
