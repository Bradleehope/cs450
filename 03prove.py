from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import metrics
import pandas as pd


def prepare_data_cars(names):
    data = pd.read_csv("car.txt", header=None, skipinitialspace=True,
                       names=names, na_values=["?"])
    data[data.isna().any(axis=1)]
    print(data.isnull().sum())
    data.dtypes
    data.dropna(inplace=True)
    print(data.isnull().sum())

    print(data.dtypes)
    return data


def prepare_data_mpg(names):
    data = pd.read_csv("auto-mpg.txt", header=None, delim_whitespace=True,
                       skipinitialspace=True,
                       names=names, na_values=["?"])
    data[data.isna().any(axis=1)]
    print(data.isnull().sum())
    data.dtypes
    data.dropna(inplace=True)
    print(data.isnull().sum())

    print(data.dtypes)
    print(data)
    return data


def prepare_data_students(names):
    student_data1 = pd.read_csv("student-mat.csv", header=0, delim_whitespace=True,
                                skipinitialspace=True,
                                na_values=["?"])
    student_data1[student_data1.isna().any(axis=1)]
    print(student_data1.isnull().sum())
    student_data1.dtypes
    student_data1.dropna(inplace=True)
    print(student_data1.isnull().sum())

    print(student_data1.dtypes)
    print(student_data1)

    student_data2 = pd.read_csv("student-por.csv", header=0, delim_whitespace=True,
                                skipinitialspace=True,
                                na_values=["?"])
    student_data2[student_data2.isna().any(axis=1)]
    print(student_data2.isnull().sum())
    student_data2.dtypes
    student_data2.dropna(inplace=True)
    print(student_data2.isnull().sum())

    print(student_data2.dtypes)
    print(student_data2)
    return student_data1


def create_cat_codes_car(data):
    data_cat = data
    data_cat.buying = data_cat.buying.astype('category')
    data_cat['buying'] = data_cat.buying.cat.codes

    data_cat.maint = data_cat.maint.astype('category')
    data_cat['maint'] = data_cat.maint.cat.codes

    data_cat.doors = data_cat.doors.astype('category')
    data_cat['doors'] = data_cat.doors.cat.codes

    data_cat.persons = data_cat.persons.astype('category')
    data_cat['persons'] = data_cat.persons.cat.codes

    data_cat.lug_boot = data_cat.lug_boot.astype('category')
    data_cat['lug_boot'] = data_cat.lug_boot.cat.codes

    data_cat.safety = data_cat.safety.astype('category')
    data_cat['safety'] = data_cat.safety.cat.codes

    data_cat.quality = data_cat.quality.astype('category')
    data_cat['quality'] = data_cat.quality.cat.codes

    print(data_cat)
    return data_cat


def create_cat_codes_mpg(data):
    data_cat = data
    data_cat.cylinders = data_cat['cylinders'].astype('category')

    data_cat.displacement = data_cat['displacement'].astype('category')
    data_cat.displacement = pd.cut(data_cat['displacement'], 10).head()

    data_cat.horsepower = data_cat['horsepower'].astype('category')
    data_cat.horsepower = pd.cut(data_cat['horsepower'], 10).head()

    data_cat.weight = data_cat['weight'].astype('category')
    data_cat.weight = pd.cut(data_cat['weight'], 10).head()

    data_cat.acceleration = data_cat['acceleration'].astype('category')
    data_cat.acceleration = pd.cut(data_cat['acceleration'], 10).head()

    data_cat.model_year = data_cat['model_year'].astype('category')

    data_cat.origin = data_cat['origin'].astype('category')

    data_cat.car_name = data_cat['car_name'].astype('category')
    data.car_name = data.car_name.cat.codes
    return data


def predict_data_car(data, X, y, data_set):
    xTrain, xTest, yTrain, yTest = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(xTrain, yTrain)
    predictions = classifier.predict(xTest)
    print("Iris Prediction Accuracy with Neighbors for ", data_set)
    print(metrics.accuracy_score(yTest, predictions)*100)


def predict_data_mpg(data, X, y, data_set):
    xTrain, xTest, yTrain, yTest = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
    regressor = KNeighborsRegressor(n_neighbors=3)
    regressor.fit(xTrain, yTrain)
    predictions = regressor.predict(xTest)
    print("Iris Prediction Accuracy with Neighbors for ", data_set)
    print(metrics.r2_score(yTest, predictions.round(), normalize=False)*100)


if __name__ == "__main__":
    car_names = ['buying', 'maint', 'doors', 'persons',
                 'lug_boot', 'safety', 'quality']
    car_data = prepare_data_cars(car_names)
    car_data_cat = create_cat_codes_car(car_data)
    car_X = car_data.drop(columns=['quality']).values
    car_y = car_data['quality'].values.flatten()
    predict_data_car(car_data_cat, car_X, car_y, "Cars:")

    mpg_names = ['mpg', 'cylinders',
                 'displacement', 'horsepower',
                 'weight', 'acceleration', 'model_year',
                 'origin', 'car_name']
    mpg_data = prepare_data_mpg(mpg_names)
    mpg_data_cat = create_cat_codes_mpg(mpg_data)
    mpg_data_cat = mpg_data_cat.drop(columns=['car_name']).values
    mpg_y = mpg_data.mpg
    mpg_X = mpg_data_cat
    predict_data_mpg(mpg_data_cat, mpg_X, mpg_y, "MPG:")
