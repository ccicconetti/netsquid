"""This module specifies classes that help with the collection of statistics.
"""

import math
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

    def scale(self, metric, value):
        """Apply a constant scale factor to all values of a metric.

        Parameters
        ----------
        metric : string
            The metric name
        value : float
            The scale factor to apply
        
        """

        if metric in self._counters:
            self._counters[metric][0] = self._counters[metric][0] * value
        elif metric in self._data:
            self._data[metric] = [x * value for x in self._data[metric]]
        else:
            raise KeyError(f'Unknown metric: {metric}')

def plot_all(x_values, xlabel, stats, block=False):
    if len(x_values) != len(stats):
        raise ValueError('Inconsistent sizes')

    nplots = len(stats[0].count_metrics()) + len(stats[0].point_metrics())

    ncols = int(math.ceil(math.sqrt(nplots)))
    nrows = 1 + (nplots - 1) // ncols
    assert ncols * nrows >= nplots

    fig, axs = plt.subplots(nrows, ncols)

    counter = 0
    count_metrics = list(stats[0].count_metrics())
    point_metrics = list(stats[0].point_metrics())
    for metric in count_metrics + point_metrics:
        xpos = counter // ncols
        ypos = counter % ncols
        if metric in count_metrics:
            plot_single(x_values, xlabel, stats, metric, Stat.get_sum, axs[xpos][ypos])
        else:
            boxplot_single(x_values, xlabel, stats, metric, axs[xpos][ypos])
        counter += 1

    fig.tight_layout()
    plt.show(block=block)

def plot_single(x_values, xlabel, stats, metric, func, ax):

    y_values = [] 
    for stat in stats:
        y_values.append(func(stat, metric))
    ax.plot(x_values, y_values, marker='o')

    ax.set(xlabel=xlabel, ylabel=metric)
    ax.grid()

def boxplot_single(x_values, xlabel, stats, metric, ax):
    y_values = [] 
    for stat in stats:
        y_values.append(stat.get_all(metric))
    ax.boxplot(y_values, positions=x_values, notch=1, sym='k+')

    ax.set(xlabel=xlabel, ylabel=metric)
    ax.grid()

    avg_values = [] 
    for stat in stats:
        avg_values.append(stat.get_avg(metric))
    ax.plot(x_values, avg_values)
