from sklearn import svm                         #SVM支持向量机
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pickle  

class SVMmodel:
    #初始化
    def __init__(self,data, result):
        self.__d_train, self.__d_test, self.__r_train, self.__r_test = train_test_split(data, result, random_state=1, train_size=0.8)
        self.__data = data
        self.__result = result

#############################################################
#Normal
#############################################################
    def classication(self, getFromFile = True):
        if getFromFile:
            f = open('svm.model','rb')
            s = f.read()
            self.__clf = pickle.loads(s)
            f.close()
        else:
            self.__clf = svm.SVC(C=0.8, kernel='rbf', gamma=20)
            self.__clf.fit(self.__d_train, self.__r_train) 
            s = pickle.dumps(self.__clf)
            f = open('svm.model', "wb+")
            f.write(s)
            f.close()
    
    def predictTestandTrain(self):
        self.__r_test_hat = self.__clf.predict(self.__d_test)
        self.__r_train_hat = self.__clf.predict(self.__d_train)
        
    def predictNewSentence(self, sentence):
        return self.__clf.predict(sentence)

    def accuracyTest(self):
        return metrics.accuracy_score(self.__r_test_hat, self.__r_test)

    def f1scoreTest(self):
        return metrics.f1_score(self.__r_test_hat, self.__r_test)

    def f1scoreTrain(self):
        return metrics.f1_score(self.__r_train_hat, self.__r_train)

##########################################################################
#Cross validation
##########################################################################
    def classicationCrossValisdation(self):
        self.__clf = svm.SVC(C=0.8, kernel='rbf', gamma=20)
        scores = cross_val_score(self.__clf, self.__data, self.__result, cv=5)  #cv为迭代次数。
        print(scores)  # 打印输出每次迭代的度量值（准确度）
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def getSVMmodel(data, result):
    return SVMmodel(data, result)