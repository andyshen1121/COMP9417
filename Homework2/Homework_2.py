import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def pre_processing(x):
    max_x = x.max()
    min_x = x.min()
    for i in range(x.size):
        x_new = float((x[i] - min_x) / (max_x - min_x))
        x[i] = x_new
    return x


if __name__ == '__main__':
    # Get data
    data = pd.read_csv('titanic.csv', dtype=float)
    # Pre-processing
    for column in data:
        data[column] = pre_processing(data[column])
    # Creating test and training sets
    y = data['Survived'].values
    X = data.drop(columns=['Survived']).values
    X_training = X[:620]
    X_test = X[620:]
    y_training = y[:620]
    y_test = y[620:]
    # Part A
    model = DecisionTreeClassifier()
    model.fit(X_training, y_training)
    y_training_predict = model.predict(X_training)
    y_test_predict = model.predict(X_test)
    print('Accuracy score for training dataset:', accuracy_score(y_training, y_training_predict))
    print('Accuracy score for test dataset:', accuracy_score(y_test, y_test_predict))
    # Part B
    auc_scores = {}
    for num in range(2, 21):
        clf = DecisionTreeClassifier(min_samples_leaf=num)
        clf.fit(X_training, y_training)
        y_score = clf.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_score)
        auc_scores[num] = auc_score
    max_auc_score = 0
    optimal_number = 2
    for key, value in auc_scores.items():
        if value > max_auc_score:
            max_auc_score = value
            optimal_number = key
    print('An optimal min_samples_leaf is', optimal_number)
    # Part C
    training_auc_scores = {}
    test_auc_scores = {}
    for number in range(2, 21):
        model = DecisionTreeClassifier(min_samples_leaf=number)
        model.fit(X_training, y_training)
        y_training_score = model.predict_proba(X_training)[:, 1]
        training_auc_score = roc_auc_score(y_training, y_training_score)
        training_auc_scores[number] = training_auc_score
        y_test_score = model.predict_proba(X_test)[:, 1]
        test_auc_score = roc_auc_score(y_test, y_test_score)
        test_auc_scores[number] = test_auc_score
    # one plot for training set
    plt.figure(1)
    plt.title('AUC score for training set')
    plt.xlabel('iterations')
    plt.ylabel('AUC score')
    plt.plot(list(training_auc_scores.keys()), list(training_auc_scores.values()))
    # one plot for test set
    plt.figure(2)
    plt.title('AUC score for test set')
    plt.xlabel('iterations')
    plt.ylabel('AUC score')
    plt.plot(list(test_auc_scores.keys()), list(test_auc_scores.values()))
    plt.show()
    # Part D
    survived_number = 0
    female_first_number = 0
    survived_famale_first = 0
    total_number = data['Survived'].size
    for row in data['Survived']:
        if row == 1:
            survived_number += 1
    # the probability of survived
    prob_s = survived_number / total_number
    for index, row in data.iterrows():
        if row['Sex'] == 1 and row['Pclass'] == 0:
            female_first_number += 1
    # the probability of people whose gender is famale and class is first
    prob_g_c = female_first_number / total_number
    for index, row in data.iterrows():
        if row['Survived'] == 1 and row['Sex'] == 1 and row['Pclass'] == 0:
            survived_famale_first += 1
    # the probability of survived people whose gender is famale and class is first
    prob_g_c_s = survived_famale_first / survived_number
    prob = (prob_g_c_s * prob_s) / prob_g_c
    print('The probability is', prob)