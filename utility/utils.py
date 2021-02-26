"""Utility methods and classes used in CtF Problem

This module contains extra features and functions frequently used in Ctf Project.
Please include the docstrings for any method or class to easily reference from Jupyter
Any pipeline or data manipulation is excluded: they are included in dataModule.py file.

Todo:
    * Finish documenting the module
    * If necessary, include __main__ in module for debuging and quick operation checking

"""
import os
import shutil
import sys

import random

import inspect
import functools

import numpy as np
import scipy.signal


def store_args(method):
    """Stores provided method args as instance attributes.

    Decorator to automatically set parameters as a class variables

    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper

def interval_flag(step, freq, name):
    """
    Returns true at the beginning of the interval.
    It is used in case where step is not incrementing uniformly.
    The method defines internal attribute based on name for instant counter.

    Parameters
    ----------------
    step : [int] 
        The current step number
    freq : [int] 
        The interval size
    name : [str] 
        Arbitrary string for counter

    Returns
    ----------------
    flag: [bool]

    """

    count = step // freq
    if not hasattr(interval_flag, name):
        setattr(interval_flag, name, count)
    if getattr(interval_flag, name) < count:
        setattr(interval_flag, name, count)
        return True
    else:
        return False

def path_create(path, override=False):
    """
    Create directory
    If override is true, remove the directory first
    """
    if override:
        if os.path.exists(path):
            shutil.rmtree(path,ignore_errors=True)
            if os.path.exists(path):
                raise OSError("Failed to remove path {}.".format(path))

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            raise OSError("Creation of the directory {} failed".format(path))

def q_retrace(reward, value_ext, gamma, is_weight):
    retrace_weight = np.minimum(1.0, is_weight)
    td_target = reward + gamma * value_ext[1:]
    value = value_ext[:-1]
    qret = value_ext[-1]
    qrets = []
    for i in range(len(is_weight) - 1, -1, -1):
        qret = reward[i] + gamma * qret
        qrets.append(qret)
        qret = (retrace_weight[i] * (qret - td_target[i])) + value[i]
    qrets.reverse()
    return qrets


def normalize(r):
    """ take 1D float numpy array and normalize it

    Args:
        r (numpy.array): list of numbers

    Returns:
        numpy.list : return normalized list

    """

    return (r - np.mean(r)) / (np.std(r) + 1e-13)  # small addition to avoid dividing by zero


def retrace(targets, behaviors, lambda_=0.2):
    """ take target and behavior policy values, and return the retrace weight

    Args:
        target (1d array float): list of target policy values in series
            (policy of target network)
        behavior (1d array float): list of target policy values in sequence
            (policy of behavior network)
        lambda_ (float): retrace coefficient

    Returns:
        weight (1D list)          : return retrace weight
    """

    weight = []
    ratio = targets / behaviors
    for r in ratio:
        weight.append(lambda_ * min(1.0, r))

    return np.array(weight)


def retrace_prod(targets, behaviors, lambda_=0.2):
    """ take target and behavior policy values, and return the cumulative product of weights

    Args:
        target (1d array float): list of target policy values in series
            (policy of target network)
        behavior (1d array float): list of target policy values in sequence
            (policy of behavior network)
        lambda_ (float): retrace coefficient

    Returns:
        weight_cumulate (1D list) : return retrace weight in cumulative-product
    """

    return np.cumprod(retrace(targets, behaviors, lambda_))


class MovingAverage:
    """MovingAverage
    Container that only store give size of element, and store moving average.
    Queue structure of container.
    """

    def __init__(self, size):
        """__init__

        :param size: number of element that will be stored in he container
        """
        from collections import deque
        self.average = 0.0
        self.size = size

        self.queue = deque(maxlen=size)

    def __call__(self):
        """__call__"""
        return self.average

    def tolist(self):
        """tolist
        Return the elements in the container in (list) structure
        """
        return list(self.queue)

    def extend(self, l: list):
        """extend

        Similar to list.extend

        :param l (list): list of number that will be extended in the deque
        """
        # Append list of numbers
        self.queue.extend(l)
        self.size = len(self.queue)
        self.average = sum(self.queue) / self.size

    def append(self, n):
        """append

        Element-wise appending in the container

        :param n: number that will be appended on the container.
        """
        s = len(self.queue)
        if s == self.size:
            self.average = ((self.average * self.size) - self.queue[0] + n) / self.size
        else:
            self.average = (self.average * s + n) / (s + 1)
        self.queue.append(n)

    def clear(self):
        """clear
        reset the container
        """
        self.average = 0.0
        self.queue.clear()
