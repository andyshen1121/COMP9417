# from https://chrisalbon.com/machine_learning/naive_bayes/multinomial_naive_bayes_classifier/
# Load libraries
import collections
import math
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# load csv file
train = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')

# convert to vectors
tfidf = TfidfVectorizer(sublinear_tf=True)

# build training/test set
X_train = tfidf.fit_transform(train['article_words'])
X_test = tfidf.transform(test['article_words'])
y_train = train['topic']
y_test = test['topic']

# compute how many articles for each topic
count_y_train = dict(collections.Counter(y_train.tolist()))
sorted_count_y_train = sorted(count_y_train.items(), key=lambda item: item[0])
print(sorted_count_y_train)

# perform over-sampling
smote = SMOTE(random_state=42)
X_resample, y_resample = smote.fit_resample(X_train, y_train)

# set parameter for LogisticRegression and training
print('LR')
clf = LogisticRegression(
    multi_class='multinomial',
    solver='saga',
    max_iter=1000
)
model = clf.fit(X_resample, y_resample)

# get result on test set and its corresponding probability
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
prob_of_each_class = clf.predict_proba(X_test)
print(prob_of_each_class.shape)

name_list = ["ARTS CULTURE ENTERTAINMENT", "BIOGRAPHIES PERSONALITIES PEOPLE", "DEFENCE", "DOMESTIC MARKETS",
             "FOREX MARKETS",
             "HEALTH", "IRRELEVANT", "MONEY MARKETS", "SCIENCE AND TECHNOLOGY", "SHARE LISTINGS", "SPORTS"]
support = [3, 15, 13, 2, 48, 14, 266, 69, 3, 7, 60]
average_precession = 0
average_recall = 0

for i in range(prob_of_each_class.shape[1]):
    # compute the expected number of candidate articles return_num
    ratio = sorted_count_y_train[i][1] / X_train.shape[0]
    return_num = min([10, math.floor(ratio * X_test.shape[0])])

    # for each topic, take top return_num articles according to its probability
    dict_of_each_class = dict(zip(range(len(prob_of_each_class[:, i])), prob_of_each_class[:, i]))
    sorted_dict_of_each_class = sorted(dict_of_each_class.items(), key=lambda item: item[1], reverse=True)
    index_we_should_return = [j[0] for j in sorted_dict_of_each_class[:return_num]]

    # find corresponding index for those candidates articles
    result = []
    for index in index_we_should_return:
        result.append(test['article_number'].iloc[index])
    print(result)

    # compute precision, recall and f1-score
    true_value = 0
    for index in index_we_should_return:
        if test['topic'].iloc[index] == name_list[i]:
            true_value += 1
    print("topic:", name_list[i], "precession", true_value / len(result), "recall", true_value / support[i])

    if true_value / len(result) == 0 and true_value / support[i] == 0:
        f1 = 0
    else:
        f1 = 2 * (true_value / len(result)) * (true_value / support[i]) / (
                    true_value / len(result) + true_value / support[i])
    print("f1", f1)

    # compute average precision and recall
    average_precession += true_value / len(result)
    average_recall += true_value / support[i]
    print()

# print out result
average_precession = average_precession / 11
average_recall = average_recall / 11
print("average_precession", average_precession, "average_recall", average_recall)
