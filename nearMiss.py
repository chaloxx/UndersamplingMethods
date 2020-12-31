#!/usr/bin/python3


# define dataset
from sklearn.datasets import make_classification
from collections import Counter
from matplotlib import pyplot
from numpy import where
from imblearn.under_sampling import NearMiss


X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)


# summarize class distribution
counter = Counter(y)


pyplot.subplot(2, 2,1)
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))

pyplot.title("Original")
pyplot.legend()


pyplot.subplot(2, 2,2)
# define the undersampling method
undersample = NearMiss(version=1, n_neighbors=3)
# transform the dataset
X, y = undersample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))

pyplot.title("N=3")
pyplot.legend()


pyplot.subplot(2, 2,3)
# define the undersampling method
undersample = NearMiss(version=1, n_neighbors=2)
# transform the dataset
X, y = undersample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))


pyplot.title("N=2")
pyplot.legend()




pyplot.show()
