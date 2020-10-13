"""This module specifies classes that help with the collection of statistics.
"""

import matplotlib
import matplotlib.pyplot as plt

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

    def get_sum(self, metric):
        if metric in self._counters:
            return self._counters[metric][0]
        elif metric in self._data:
            return sum(self._data[metric])
        raise KeyError(f'Unknown metric {metric}')

    def get_count(self, metric):
        if metric in self._counters:
            return self._counters[metric][1]
        elif metric in self._data:
            return len(self._data[metric])
        return 0

    def get_avg(self, metric):
        if metric in self._counters:
            return self._counters[metric][0] / self._counters[metric][1]
        elif metric in self._data:
            return sum(self._data[metric]) / len(self._data[metric])
        raise KeyError(f'Unknown metric {metric}')

    def get_all(self, metric):
        if metric in self._counters:
            return self._counters[metric][0]
        elif metric in self._data:
            return self._data[metric]
        raise KeyError(f'Unknown metric {metric}')

    def count_metrics(self):
        """Return all the count metrics."""

        return set(self._counters.keys())

    def point_metrics(self):
        """Return all the point metrics."""

        return set(self._data.keys())

    def count(self, metric, value):
        if metric in self._data:
            raise KeyError(f'Metric {metric} already used')
        if metric not in self._counters:
            self._counters[metric] = [0, 0]
        record = self._counters[metric]
        record[0] += value
        record[1] += 1

    def add(self, metric, value):
        if metric in self._counters:
            raise KeyError(f'Metric {metric} already used')
        if metric not in self._data:
            self._data[metric] = []
        self._data[metric].append(value)

def plot_single(x_values, x_label, stats, metric, func, block=True):
    if len(x_values) != len(stats):
        raise ValueError('Inconsistent sizes')

    _, ax = plt.subplots()
    y_values = [] 
    for stat in stats:
        y_values.append(func(stat, metric))
    ax.plot(x_values, y_values, marker='o')

    ax.set(xlabel=x_label, ylabel=metric)
    ax.grid()

    plt.show(block=block)

def boxplot_single(x_values, x_label, stats, metric, block=True):
    if len(x_values) != len(stats):
        raise ValueError('Inconsistent sizes')

    _, ax = plt.subplots()
    y_values = [] 
    for stat in stats:
        y_values.append(stat.get_all(metric))
    ax.boxplot(y_values, positions=x_values, notch=True)

    ax.set(xlabel=x_label, ylabel=metric)
    ax.grid()

    avg_values = [] 
    for stat in stats:
        avg_values.append(stat.get_avg(metric))
    ax.plot(x_values, avg_values)

    plt.show(block=block)