from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
xTrain, xTest, yTrain, yTest = train_test_split(iris.data,
                                                iris.target,
                                                test_size=0.3)
scaler = StandardScaler()
scaler.fit(xTrain)
xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)

if __name__ == "__main__":
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(xTrain, yTrain)
    predictions = classifier.predict(xTest)
    print("Iris Prediction Accuracy with Neighbors: ", (metrics.accuracy_score(yTest, predictions))*100)
    print(predictions)
