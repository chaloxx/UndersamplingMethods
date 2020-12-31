#!/usr/bin/python3


# Undersample and plot imbalanced dataset with Tomek Links
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import TomekLinks
from matplotlib import pyplot
from numpy import where,setdiff1d,array



pyplot.subplot(1,2,1)
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# summarize class distribution
counter = Counter(y)
print(counter)

for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.title("Original")



pyplot.subplot(1,2,2)
# define the undersampling method
undersample = TomekLinks()
# transform the dataset
newX, y = undersample.fit_resample(X, y)
deleteX = array([point for point in X if point not in newX])
print(deleteX.shape)
print(X.shape)
print(newX.shape)
# summarize the new class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(newX[row_ix, 0], newX[row_ix, 1], label=str(label))


pyplot.scatter(deleteX[:,0],deleteX[:,1],label="Deleted")
pyplot.legend()
pyplot.title("Resampling with Tomek Links")
pyplot.show()
