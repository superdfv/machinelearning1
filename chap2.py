
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#generate data
# Z, w = mglearn.datasets.make_forge()

#plot

# mglearn.discrete_scatter(Z[:, 0], Z[:, 1], w)
# plt.legend(['Class 0', 'Class 1'], loc=4)
# plt.xlabel('First feature')
# plt.ylabel('Second feature')
# print ('X.shape: {}'.format(Z.shape))

#regression algo

# X, y = mglearn.datasets.make_wave(n_samples=40)
# plt.plot(X, y, 'o')
# plt.ylim(-3, 3)
# plt.xlabel("Feature")
# plt.ylabel("Target")
# plt.show()

#cancer example

cancer = load_breast_cancer()
print('cancer.keys():\n{}'.format(cancer.keys()))
print('Shape of cancer data:{}'.format(cancer.data.shape))
print("Sample counts per class:\n{}".format(
        {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("Feature names:\n{}".format(cancer.feature_names))

#boston example

boston = load_boston()
print('boston.keys():\n{}'.format(boston.keys()))
print("Data shape: {}".format(boston.data.shape))

# product of crime rate and highway accessibility
# X, y = mglearn.datasets.load_extended_boston()
# print("X.shape: {}".format(X.shape))


#knearest neighbors plot

# mglearn.plots.plot_knn_classification(n_neighbors=5)
# plt.show()

#apply  k-nearest whit scikit-learn

# X, y = mglearn.datasets.make_forge()
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# clf = KNeighborsClassifier(n_neighbors=3)
# clf.fit(X_train, y_train)
# print("Test set predictions: {}".format(clf.predict(X_test)))
# print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

#plot decision boundaries fro 1,3 & 9 neighbors

# fig, axes = plt.subplots(1, 3, figsize=(10, 3))
# for n_neighbors, ax in zip([1, 3, 9], axes):
#     # the fit method returns the object self, so we can instantiate and fit in one line
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
#     mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#     ax.set_title("{} neighbor(s)".format(n_neighbors))
#     ax.set_xlabel("feature 0")
#     ax.set_ylabel("feature 1")
# axes[0].legend(loc=3)
# plt.show()

#cancer data with diferent neighbors

X_train, X_test, y_train, y_test = train_test_split(
                                    cancer.data, cancer.target, stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10

neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

