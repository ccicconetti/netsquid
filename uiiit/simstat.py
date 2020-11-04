"""This module specifies classes that help with the collection of statistics.
"""

import json
import math
import matplotlib
import matplotlib.pyplot as plt
import os
import re
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

    def __contains__(self, key):
        return key in self._params.keys()

    def change_param(self, key, value):
        self._params[key] = value
    
    def del_param(self, key):
        try:
            del self._params[key]
        except:
            pass

    def all_params(self):
        return self._params

    def match(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._params:
                raise KeyError(f'Invalid parameter: {k}')
            if self._params[k] != v:
                return False
        return True

    def compact(self, keys=None):
        return '.'.join([f'{k}={v}' for k, v in sorted(self._params.items()) if keys is None or k in keys])

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

    def __init__(self, conf=None):
        self._conf = conf if conf is not None else Conf()
        self._points = dict()
        self._counts = dict()

    def __repr__(self):
        """Return a string representation of the parameters."""

        return str(self._conf)

    def __eq__(self, other):
        """Two `Stat` objects are equal if they have the same parameters."""

        if other is None:
            return False

        return self._conf.all_params() == other._conf.all_params()

    def __contains__(self, metric):
        """Return True if the metric is in this object."""

        if metric in self._counts.keys() or metric in self._points.keys():
            return True
        return False

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

    def get_avg_ci(self, metric):
        if metric in self._points:
            if len(self._points[metric]) <= 2:
                return (self.get_avg(metric), 0)
            stat = sms.DescrStatsW(self._points[metric])
            ci = stat.tconfint_mean(alpha=0.05)
            return ((ci[1] + ci[0]) / 2, ci[1] - ci[0])
        return (self.get_sum(metric), 0)

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

    def export(self, path, parameters=None):
        """Export the content to a set of files in the given path.
        
        Parameters
        ----------
        path : str
            The path where to save the files.
        parameters : list, optional
            The list of parameters to use for the file name.
            If empty then use all.
        
        """

        base_name = self._conf.compact(parameters)
        if base_name:
            base_name += '.'

        for metric_name, record in self._counts.items():
            with open(f'{path}/{base_name}{metric_name}.dat', 'w') as outfile:
                outfile.write(f'{record[0]} {record[1]}\n')

        for metric_name, values in self._points.items():
            with open(f'{path}/{base_name}{metric_name}.dat', 'w') as outfile:
                for value in values:
                    outfile.write(f'{value}\n')

    def merge(self, regex, outname):
        """Merge all the values of a number of metrics into a new one.

        Parameters
        ----------
        regex : str
            The regular expression to decide which metrics to match.
            If no metrics match then the new metric is not added, but no
            exception is raised.
        outname : str
            The name of the output metric.

        Returns
        -------
        `Stat`
            This object.
        
        Raises
        ------
        KeyError
            If `regex` matches a mix of point and count metrics.
        ValueError
            If `outname` already exists.
        
        """

        if outname in self:
            raise ValueError(f'Cannot overwrite metric: {outname}')

        newrecord = None
        for metric, record in self._counts.items():
            if re.match(regex, metric):
                if newrecord is None:
                    newrecord = [0, 0]
                newrecord[0] += record[0]
                newrecord[1] += record[1]
        
        newpoints = []
        for metric, values in self._points.items():
            if re.match(regex, metric):
                newpoints += values

        if newrecord is not None and newpoints:
            raise KeyError(f'Regex matches both count and point metrics: {regex}')

        if newrecord is not None:
            self._counts[outname] = newrecord

        if newpoints:
            self._points[outname] = newpoints

        return self

    def add_avg(self, metric, name=None):
        """Add a new metric that is the average of the given one.

        Parameters
        ----------
        metric : str
            The name of the metric for which to compute the average.
        name : str, optional
            The name of the new metric. If None, then the name is given
            by `metric` with -avg appended.

        Returns
        -------
        `Stat`
            This object.

        Raises
        ------
        KeyError
            If `metric` does not exist.
        ValueError
            If `name` is already existing.

        """

        newmetric = f'{metric}-avg' if name is None else name
        if newmetric in self:
            raise ValueError(f'Cannot overwrite existing metric: {newmetric}')

        self.count(newmetric, self.get_avg(metric))

        return self

    def print(self, metrics=None):
        """Print the content to human-readable format.
        
        Parameters
        ----------
        metrics : None or iterable
            Print only this metrics. If None print all.
        """

        for metric, record in self._counts.items():
            if metrics is None or metric in metrics:
                print(f"{metric} = {record[0]}")
        for metric, values in self._points.items():
            if metrics is None or metric in metrics:
                print(f"{metric} = {[f'{x:.3f}' for x in values]}")

class MultiStat:
    """A collection of Stat objects that can be serialized/deserialized.

    """
    def __init__(self, initial=None):
        self._stats = dict()
        if initial:
            for stat in initial:
                self.add(stat)

    def _add_single(self, stat):
        """Add a `Stat` object to the collection doing nothing if present."""

        if not isinstance(stat, Stat):
            raise TypeError('Expected a Stat object when calling _add_single()')

        key = str(stat)
        if key in self._stats:
            return False
        self._stats[key] = stat
        return True

    def add(self, stats):
        """Add one or more `Stat` objects to the collection.
        
        Do nothing if any of them are already present.

        Parameters
        ----------
        stats : `Stat` or list of `Stat` objects
            The item to add to the collection.
        
        Returns
        -------
        True if at least one object has been added, False otherwise.

        """

        if hasattr(stats, '__iter__'):
            ret = False
            for stat in stats:
                ret |= self._add_single(stat)
            return ret

        return self._add_single(stats)

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

    def all_values(self):
        """Return all the `Stat` objects stored in the collection as a list."""

        return list(self._stats.values())

    def get_stats(self, **params):
        """Return a list of `Stat` objects matching the given parameters."""

        stats = []
        for stat in self._stats.values():
            if stat.conf().match(**params):
                stats.append(stat)

        return stats

    def __len__(self):
        """Return the number of elements stored in the container."""

        return len(self._stats)

    def empty(self):
        """Return True if there are not items in the collection."""

        return True if not self._stats else False

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

    def filter(self, **kwargs):
        """Return a `MultiStat` objects with elements matching the given criteria.

        If a parameter is not specified, then it is assumed that any value is OK.

        """

        stats = []
        for stat in self._stats.values():
            if stat.conf().match(**kwargs):
                stats.append(stat)
        return MultiStat(stats)

    def param_values(self, param):
        """Return all the values for the given parameter.
        
        Raises
        ------
        KeyError
            If the given parameter is not specified in at least one element.
        """

        ret = set()
        for stat in self._stats.values():
            ret.add(stat.conf()[param])
        return ret

    def apply_to_all(self, func):
        """Apply a given function to all the `Stat` objects in the collection."""

        for stat in self._stats.values():
            func(stat)

        return self

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
        """Deserialize from a given file. Convenience wrapper of `json_load`.

        If `path` does not exist then an empty `MultiStat` is returned.
        
        Parameters
        ----------
        path : str
            The path of the file to open.

        Returns
        -------
        A `MultiStat` object initialized with the content from the JSON file.
        """

        try:
            with open(path, 'r') as infile:
                return MultiStat.json_load(infile)
        except FileNotFoundError:
            return MultiStat()

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

        self._create_dir(path)
        for stat in self._stats.values():
            stat.export(path, self._variable_params())

    def print(self):
        """Print all the `Stat` objects in a human-readable manner."""

        for stat in self._stats.values():
            print(f'** {stat}')
            stat.print()

    def single_factor_export(self, factor, path):
        """Export all the average and confidence intervals of a factor.

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

        self._create_dir(path)
        data = self.single_factor_data(factor)
        for metric, per_metric in data.items():
            for mangle, per_mangle in per_metric.items():
                if not mangle:
                    basename = ''
                else:
                    basename = f'{mangle}.'
                with open(path + f'/{basename}{metric}.dat', 'w') as outfile:
                    for factor_value in sorted(per_mangle.keys()):
                        (avg, ci) = per_mangle[factor_value]
                        outfile.write(f'{factor_value} {avg} {ci}\n')

    def single_factor_data(self, factor):
        variable_params = set(self._variable_params())
        variable_params.remove(factor)
        if variable_params is None:
            variable_params = set()
        data = dict()
        for stat in self._stats.values():
            factor_value = stat.conf()[factor]
            mangle = stat.conf().compact(variable_params)
            for metric in list(stat.count_metrics()) + list(stat.point_metrics()):
                if metric not in data:
                    data[metric] = dict()
                if mangle not in data[metric]:
                    data[metric][mangle] = dict()
                data[metric][mangle][factor_value] = stat.get_avg_ci(metric)
        return data

    @staticmethod
    def _create_dir(path):
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise FileExistsError(f'Path {path} exists but it is not a directory')
        else:
            os.mkdir(path)

    def _variable_params(self):
        """Return all parameters that have the same value across all the stats."""

        all_params = dict()
        for stat in self._stats.values():
            for k, v in stat.conf().all_params().items():
                if k not in all_params:
                    all_params[k] = set()
                all_params[k].add(v)
        for stat in self._stats.values():
            for param in all_params.keys():
                if param not in stat.conf():
                    all_params[param].add(None)
        variable_params = []
        for param, values in all_params.items():
            if len(values) > 1:
                variable_params.append(param)
        return variable_params

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
