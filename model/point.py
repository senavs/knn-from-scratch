import numpy as np


class Point:

    def __init__(self, axis):
        """Point constructor

        :param axis: iterable with point coordinates
            :type list, np.array and tuple:
            :example:
                        x    y    z     w
                axis = [1, 0.3, 6.4, -0.2]
        """
        self.axis = np.array(axis)

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.axis + other.axis)
        return Point(self.axis + np.array(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.axis - other.axis)
        return Point(self.axis - np.array(other))

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, Point):
            return Point(self.axis * other.axis)
        return Point(self.axis * np.array(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Point):
            return Point(self.axis / other.axis)
        return Point(self.axis / np.array(other))

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __floordiv__(self, other):
        if isinstance(other, Point):
            return Point(self.axis // other.axis)
        return Point(self.axis // np.array(other))

    def __rfloordiv__(self, other):
        return self.__floordiv__(other)

    def __pow__(self, power, modulo=None):
        if modulo:
            return self.axis ** power % modulo
        return self.axis ** power

    def __eq__(self, other):
        if isinstance(other, Point):
            return max(self.axis == other.axis)
        return max(self.axis == other)

    def __getitem__(self, item):
        return self.axis[item]

    def __repr__(self):
        return f'Point{tuple(self.axis)}'
