# KNN from scratch
A Python implementation of KNN machine learning algorithm.

## Algorithm
[K nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) is a supervised learning algorithm to classification or regression. Considering _n_ points in the cartesian plane, if a new point is placed, its label will be the label of the _k_ nearest neighbors, in other words, the neighbors with least distance. To calculate the distance [euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) algorithm is used.

## Euclidean distance
<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT-BNnXJs2WFM-hvledmFTsECBmQ1ssxkLnucrp3sG8yrXA8VAN" width=250>
</p>

## Implementation
[Point](https://github.com/senavs/knn-from-scratch/blob/master/model/point.py) is a class to represent a point in cartesian plane. You are able to sum, subtract, multiply, divide and calculate distance between two points.
``` python
from model.point import Point

p1 = Point([7, 4, 3])
p2 = Point([17, 6, 2])
```
[KNearestNeighbors](https://github.com/senavs/knn-from-scratch/blob/master/model/knn.py) is the model class. Only the methods are allowed: `fit` and `predict`. Look into `help(KNearestNeighbors)` for more infomraiton.
```python
from model.knn import KNearestNeighbors

knn = KNearestNeighbors(k=3)
knn.fit(x_train, y_train)

predict = knn.predict(x_predict)
```

## Apply KNearestNeighbors from scratch in dataset
To show the package working, I create a jupyter notebook with [iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html). Take a look into [here](https://github.com/senavs/knn-from-scratch/blob/master/notebook/knn-iris_dataset.ipynb).
