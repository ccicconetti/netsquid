"""This module specifies classes that help with the collection of statistics.
"""

import math
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

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

def plot_all(x_values, xlabel, stats, metrics, block=False):
    if len(x_values) != len(stats):
        raise ValueError('Inconsistent sizes')

    all_metrics = metrics if metrics \
        else sorted(list(stats[0].count_metrics()) + list(stats[0].point_metrics()))

    nplots = len(all_metrics)

    ncols = int(math.ceil(math.sqrt(nplots)))
    nrows = 1 + (nplots - 1) // ncols
    assert ncols * nrows >= nplots

    fig, axs = plt.subplots(nrows, ncols, squeeze=False)

    counter = 0
    for metric in all_metrics:
        xpos = counter // ncols
        ypos = counter % ncols
        if metric in stats[0].count_metrics():
            plot_single(x_values, xlabel, stats, metric, Stat.get_sum, axs[xpos][ypos])
        else:
            boxplot_single(x_values, xlabel, stats, metric, axs[xpos][ypos])
        counter += 1

    fig.tight_layout()
    plt.show(block=block)

def plot_all_same(x_values, xlabel, ylabel, stats, metrics, block=False):
    if not metrics:
        raise ValueError('Empty set of metrics to plot')
    if len(x_values) != len(stats):
        raise ValueError('Inconsistent sizes')

    _, ax = plt.subplots()

    if metrics[0] in stats[0].count_metrics():
        for metric in metrics:
            plot_single(x_values, xlabel, stats, metric, Stat.get_sum, ax)
        ax.set_ylabel(ylabel)
        ax.grid()
    else:
        plot_multi(x_values, xlabel, ylabel, stats, metrics, ax)

    plt.legend(loc='best')

    plt.show(block=block)

def plot_multi(x_values, xlabel, ylabel, stats, metrics, ax):
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid()

    for metric in sorted(metrics):
        x_values_with_data = []
        y_values = []
        e_values = []
        for stat,x_value in zip(stats, x_values):
            if metric not in stat.point_metrics():
                continue
            x_values_with_data.append(x_value)
            y_values.append(stat.get_avg(metric))
            if stat.get_count(metric) >= 2:
                stat = sms.DescrStatsW(stat.get_all(metric))
                ci = stat.tconfint_mean(alpha=0.05)
                e_values.append(ci[1] - ci[0])
            else:
                e_values.append(0)

        ax.errorbar(x_values_with_data, y_values, e_values, marker='x', label=metric, capsize=5)

def plot_single(x_values, xlabel, stats, metric, func, ax):
    ax.set(xlabel=xlabel, ylabel=metric)
    ax.grid()

    y_values = [] 
    for stat in stats:
        y_values.append(func(stat, metric))

    ax.plot(x_values, y_values, marker='x', label=metric)

def boxplot_single(x_values, xlabel, stats, metric, ax):
    ax.set(xlabel=xlabel, ylabel=metric)
    ax.grid()

    y_values = []
    x_values_with_data = []
    avg_values = [] 
    for stat, x_value in zip(stats, x_values):
        if metric in stat.point_metrics():
            y_values.append(stat.get_all(metric))
            x_values_with_data.append(x_value)
            avg_values.append(stat.get_avg(metric))

    ax.boxplot(y_values, positions=x_values_with_data, notch=1, sym='k+')
    ax.plot(x_values_with_data, avg_values, label=metric)
