from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from random import shuffle


iris = datasets.load_iris()
# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)

xTrain, xTest, yTrain, yTest = train_test_split(iris.data, iris.target, test_size = 0.3)
print("xtrain: ", xTrain)
print("xtest: ", xTest)
print("ytrain: ", yTrain)
print("ytest: ", yTest)

classifier = GaussianNB()
classifier.fit(xTrain, yTrain)

targets_predicted = classifier.predict(xTest)
print("Iris Prediction Accuracy: ", (metrics.accuracy_score(yTest, targets_predicted))*100)

class HardCodedClassifier(object):
    def fit(self, dataset, target):
        print("I am learning")
        
    def predict(self, dataset):
        predict = []
        for row in dataset:
            predict.append(0)
        return predict

if __name__ == "__main__":
    classifier = HardCodedClassifier()
    classifier.fit(iris.data, 0)
    prediction = classifier.predict(xTest)
    expected_predict = []
    for row in xTest:
        expected_predict.append(0)
    
    print("Iris Prediction Accuracy with Hard Code: ", (metrics.accuracy_score(expected_predict, prediction))*100)
    