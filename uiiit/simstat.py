"""This module specifies classes that help with the collection of statistics.
"""

import json
import math
import matplotlib
import matplotlib.pyplot as plt
import os
import statsmodels.stats.api as sms

__all__ = [
    "Conf",
    "Stat",
    "MultiStat",
    ]

class Conf:
    """Experiment configuration."""
    def __init__(self, **kwargs):
        self._params = kwargs

    def __getitem__(self, key):
        return self._params[key]

    def change_param(self, key, value):
        self._params[key] = value
    
    def del_param(self, key):
        try:
            del self._params[key]
        except:
            pass

    def all_params(self):
        return self._params

    def compact(self):
        return '.'.join([f'{k}={v}' for k, v in sorted(self._params.items())])

    def __repr__(self):
        return ', '.join([f'{k}: {v}' for k, v in sorted(self._params.items())])

class Stat:
    """A simple class holding count/point metrics. An object of this class
    is also assigned a given configuration, which is supposed to identify it
    uniquely within the analysis.

    Parameters
    ----------
    conf : `Conf`
        The configuration identifying this statistics object.
    
    """

    def __init__(self, **params):
        self._conf = Conf(**params)
        self._points = dict()
        self._counts = dict()

    def __repr__(self):
        """Return a string representation of the parameters."""

        return str(self._conf)

    def __eq__(self, other):
        """Two `Stat` objects are equal if they have the same parameters."""

        return self._conf.all_params() == other._conf.all_params()

    def conf(self):
        """Return the conf object."""

        return self._conf

    def change_param(self, key, value):
        """Change/add the value of a parameter."""

        self._conf.change_param(key, value)
    
    def del_param(self, key):
        """Remove a parameter. Ignore if it does not exist."""

        self._conf.del_param(key)

    def get_sum(self, metric):
        if metric in self._counts:
            return self._counts[metric][0]
        elif metric in self._points:
            return sum(self._points[metric])
        raise KeyError(f'Unknown metric {metric}')

    def get_count(self, metric):
        if metric in self._counts:
            return self._counts[metric][1]
        elif metric in self._points:
            return len(self._points[metric])
        return 0

    def get_avg(self, metric):
        if metric in self._counts:
            return self._counts[metric][0] / self._counts[metric][1]
        elif metric in self._points:
            return sum(self._points[metric]) / len(self._points[metric])
        raise KeyError(f'Unknown metric {metric}')

    def get_all(self, metric):
        if metric in self._counts:
            return self._counts[metric][0]
        elif metric in self._points:
            return self._points[metric]
        raise KeyError(f'Unknown metric {metric}')

    def count_metrics(self):
        """Return all the count metrics."""

        return set(self._counts.keys())

    def point_metrics(self):
        """Return all the point metrics."""

        return set(self._points.keys())

    def count(self, metric, value):
        if metric in self._points:
            raise KeyError(f'Metric {metric} already used')
        if metric not in self._counts:
            self._counts[metric] = [0, 0]
        record = self._counts[metric]
        record[0] += value
        record[1] += 1

    def add(self, metric, value):
        if metric in self._counts:
            raise KeyError(f'Metric {metric} already used')
        if metric not in self._points:
            self._points[metric] = []
        self._points[metric].append(value)

    def scale(self, metric, value):
        """Apply a constant scale factor to all values of a metric.

        Parameters
        ----------
        metric : string
            The metric name
        value : float
            The scale factor to apply
        
        """

        if metric in self._counts:
            self._counts[metric][0] = self._counts[metric][0] * value
        elif metric in self._points:
            self._points[metric] = [x * value for x in self._points[metric]]
        else:
            raise KeyError(f'Unknown metric: {metric}')

    def content_dump(self):
      """Return the content of this object, e.g., for serialization."""

      return {
          'conf' : self._conf.all_params(),
          'points' : self._points,
          'counts' : self._counts,
          }

    @staticmethod
    def make_from_content(params, points, counts):
        """Create and return a new `Stat` object with the given internal state."""

        ret = Stat()
        ret._conf = Conf(**params)
        ret._points = points
        ret._counts = counts
        return ret

    def export(self, path):
        """Export the content to a set of files in the given path."""

        base_name = self._conf.compact()

        for metric_name, record in self._counts.items():
            with open(f'{path}/{base_name}-{metric_name}.dat', 'w') as outfile:
                outfile.write(f'{record[0]} {record[1]}\n')

        for metric_name, values in self._points.items():
            with open(f'{path}/{base_name}-{metric_name}.dat', 'w') as outfile:
                for value in values:
                    outfile.write(f'{value}\n')

class MultiStat:
    """A collection of Stat objects that can be serialized/deserialized.

    """
    def __init__(self, initial=None):
        self._stats = dict()
        if initial:
            for stat in initial:
                self.add(stat)

    def add(self, stat):
        """Add a stat object to the collection. Do nothing if already present.

        Parameters
        ----------
        stat : `Stat`
            The item to add to the collection.
        
        Returns
        -------
        True if added, False otherwise.

        """

        key = str(stat)
        if key in self._stats:
            return False
        self._stats[key] = stat
        return True

    def __getitem__(self, conf):
        """Return the `Stat` object associated to the given configuration.

        Parameters
        ----------
        conf : `Conf`
            The configuration for which the caller wishes to retrieve the `Stat`.
        
        Returns
        -------
        The `Stat` object associated to the given `Conf`, if any.

        Raises
        ------
        KeyError
            If there are no statistics associated to the given `Conf`.
        
        """

        return self._stats[str(conf)]

    def __contains__(self, conf):
        """Return True if the configuration is already in the collection."""

        return str(conf) in self._stats

    def all_confs(self):
        """Return all the configurations stored in the collection."""

        return set(self._stats.keys())

    def __len__(self):
        """Return the number of elements stored in the container."""

        return len(self._stats)

    def json_dump(self, fp):
        """Serialize the content of the collection."""

        data = []
        for stat in self._stats.values():
            data.append(stat.content_dump())

        json.dump(data, fp)

    def json_dump_to_file(self, path):
        """Serialize the content of the collection to a file. Convenience wrapper of `json_dump`."""

        with open(path, 'w') as outfile:
            self.json_dump(outfile)

    @staticmethod
    def json_load(fp):
        """Deserialize from the given stream to a new `MultiStat` object."""

        mstat = MultiStat()
        for content in json.load(fp):
            mstat.add(Stat.make_from_content(
                params=content['conf'],
                points=content['points'],
                counts=content['counts']))
        return mstat

    @staticmethod
    def json_load_from_file(path):
        """Deserialize from a given file. Convenience wrapper of `json_load`."""

        with open(path, 'r') as infile:
            return MultiStat.json_load(infile)

    def export(self, path):
        """Export all the `Stat` in the collection to text files.

        Parameters
        ----------
        path : str
            The directory that will store the files. If it does not exist,
            it will be created. If it exists but it is not a directory, an
            exception will be raised.

        Raises
        ------
        FileExistsError
            If the path exists but it is not a directory
        """

        if os.path.exists(path):
            if not os.path.isdir(path):
                raise FileExistsError(f'Path {path} exists but it is not a directory')
        else:
            os.mkdir(path)

        for stat in self._stats.values():
            stat.export(path)

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
        x_values_with_points = []
        y_values = []
        e_values = []
        for stat,x_value in zip(stats, x_values):
            if metric not in stat.point_metrics():
                continue
            x_values_with_points.append(x_value)
            y_values.append(stat.get_avg(metric))
            if stat.get_count(metric) >= 2:
                stat = sms.DescrStatsW(stat.get_all(metric))
                ci = stat.tconfint_mean(alpha=0.05)
                e_values.append(ci[1] - ci[0])
            else:
                e_values.append(0)

        ax.errorbar(x_values_with_points, y_values, e_values, marker='x', label=metric, capsize=5)

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
    x_values_with_points = []
    avg_values = [] 
    for stat, x_value in zip(stats, x_values):
        if metric in stat.point_metrics():
            y_values.append(stat.get_all(metric))
            x_values_with_points.append(x_value)
            avg_values.append(stat.get_avg(metric))

    ax.boxplot(y_values, positions=x_values_with_points, notch=1, sym='k+')
    ax.plot(x_values_with_points, avg_values, label=metric)
