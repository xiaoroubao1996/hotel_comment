from sklearn import svm                         #SVM支持向量机
from sklearn import metrics
from sklearn.model_selection import train_test_split

class SVMmodel:
    #初始化
    def __init__(self,data, result):
        self.__d_train, self.__d_test, self.__r_train, self.__r_test = train_test_split(data, result, random_state=1, train_size=0.8)

    def classication(self):
        self.__clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
        self.__clf.fit(self.__d_train, self.__r_train) 
        # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        #     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        #     max_iter=-1, probability=False, random_state=None, shrinking=True,
        #     tol=0.001, verbose=False)
    
    def predict(self):
        self.__r_hat = self.__clf.predict(self.__d_test)
        

    def accuracy(self):
        return metrics.accuracy_score(self.__r_hat, self.__r_test)

    def f1score(self):
        return metrics.f1_score(self.__r_hat, self.__r_test)

def getSVMmodel(data, result):
    return SVMmodel(data, result)