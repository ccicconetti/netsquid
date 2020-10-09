"""This module specifies classes that help with the collection of statistics.
"""

__all__ = [
    "Stat",
    ]

class Stat:
    def __init__(self, **params):
        self._params = params
        self._data = dict()
        self._counters = dict()

    def __repr__(self):
        return ', '.join([f'{k}: {v}' for k, v in self._params.items()])

    def get_count_sum(self, metric):
        return self._counters[metric][0]

    def get_count_avg(self, metric):
        return self._counters[metric][0] / self._counters[metric][1]

    def count(self, metric, value):
        if metric not in self._counters:
            self._counters[metric] = [0, 0]
        record = self._counters[metric]
        record[0] += value
        record[1] += 1

    def add(self, metric, value):
        if metric not in self._data:
            self._data[metric] = []
        self._data[metric].append(value)

    def get_add(self, metric):
        return self._data[metric]