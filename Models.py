import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt


# data_set =  pd.read_csv("usa_movies_processed.csv")
data_set =  pd.read_json("usa_movies_processed.json", lines=True)
y = data_set.genre
X = data_set.drop(["genre"], axis=1)


import numpy as np
from sklearn.model_selection import StratifiedKFold,train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


pipe_lr = Pipeline([('scl', StandardScaler()),('clf', LogisticRegression(penalty='l2', random_state=0))])






from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
              {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)




#
# #VALIDATION CURVE
#
# param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# train_scores, test_scores = validation_curve(
#                 estimator=pipe_lr,
#                 X=X_train,
#                 y=y_train,
#                 param_name='clf__C',
#                 param_range=param_range,
#                 cv=10, n_jobs=-1)
#
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
#
# plt.plot(param_range, train_mean,
#          color='blue', marker='o',
#          markersize=5, label='training accuracy')
#
# plt.fill_between(param_range, train_mean + train_std,
#                  train_mean - train_std, alpha=0.15,
#                  color='blue')
#
# plt.plot(param_range, test_mean,
#          color='green', linestyle='--',
#          marker='s', markersize=5,
#          label='validation accuracy')
#
# plt.fill_between(param_range,
#                  test_mean + test_std,
#                  test_mean - test_std,
#                  alpha=0.15, color='green')
#
# plt.grid()
# plt.xscale('log')
# plt.legend(loc='lower right')
# plt.xlabel('Parameter C')
# plt.ylabel('Accuracy')
# plt.ylim([0.1, 0.5])
# plt.tight_layout()
# # plt.savefig('./figures/validation_curve.png', dpi=300)
# plt.show()
#










# LEARNING CURVE


# train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X = X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=-1)
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
#
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
#
#
# plt.plot(train_sizes, train_mean,
#          color='blue', marker='o',
#          markersize=5, label='training accuracy')
#
# plt.fill_between(train_sizes,
#                  train_mean + train_std,
#                  train_mean - train_std,
#                  alpha=0.15, color='blue')
#
# plt.plot(train_sizes, test_mean,
#          color='green', linestyle='--',
#          marker='s', markersize=5,
#          label='validation accuracy')
#
# plt.fill_between(train_sizes,
#                  test_mean + test_std,
#                  test_mean - test_std,
#                  alpha=0.15, color='green')
#
# plt.grid()
# plt.xlabel('Number of training samples')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.ylim([0.2, 0.4])
# plt.tight_layout()
# # plt.savefig('./figures/learning_curve.png', dpi=300)
# plt.show()
#





# pipe_lr.fit(X_train, y_train)
# print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
# print('Train Accuracy: %.3f' % pipe_lr.score(X_train, y_train))






























# scores = []
# kfold = StratifiedKFold(n_splits=10)
#
# for train_index, validation_index in kfold.split(X_train, y_train):
#     X_train_i, X_test_i = X_train.iloc[train_index], X_train.iloc[validation_index]
#     y_train_i, y_test_i = y_train.iloc[train_index], y_train.iloc[validation_index]
#
#
#     pipe_lr.fit(X_train_i, y_train_i)
#
#
#     score = pipe_lr.score(X_train_i, y_train_i)
#     scores.append(score)
#     print('Acc: %.3f' % (score))






temp = 10