from operator import itemgetter

from model.point import Point


class KNearestNeighbors:

    def __init__(self, k=3):
        """KNearestNeighbors constructor

        :param k: total of neighbors
            :type: int
        """
        self.k = int(k)
        self._fit_data = []

    def fit(self, x, y):
        """Train knn model with x data

        :param x: data with coordinates
            :type: list, tuple, np.array
            :example:
                [[1, 2], [2, 3], [3, 4], [4, 5]]
        :param y: labels to x data set
            :type: list, tuple, np.array
            :example:
                [[1], [0], [0], [1]]
        :return: None
        """

        assert len(x) == len(y)
        # [(Point(1, 2), [0]), (Point(0, -1), [0]), (Point(5, 5), [1])]
        self._fit_data = [(Point(coordinates), label) for coordinates, label in zip(x, y)]

    def predict(self, x):
        """Predict x array

        :param x: data with coordinates to be predicted
            :type: list, tuple, np.array
            :example:
                [[1, 2], [2, 3], [3, 4], [4, 5]]
        :return: x predicts
            :type: list
            :example:
                [[1], [0], [0], [1]]
        """

        predicts = []
        for coordinates in x:
            predict_point = Point(coordinates)

            # euclidean distance from predict_point to all in self._fit_data
            distances = []
            for data_point, data_label in self._fit_data:
                distances.append((predict_point.distance(data_point), data_label))

            # k points with less distances
            distances = sorted(distances, key=itemgetter(0))[:self.k]
            # label of k points with less distances
            predicts.append(list(max(distances, key=itemgetter(1))[1]))

        return predicts
