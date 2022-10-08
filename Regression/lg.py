#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finalized on 10:34:50, (Thu.) Dec. 16th, 2021

@filename:    lg.py
@modifier:    Chien-cheng (Jeff) Chen
@description: Main program for linear regression
              Used in linear algebra (2021) HW5
"""
# 1. ------------------- libraries ------------------- #
import numpy as np  # NOTE: no other libraies allowed!

# for type hint
from typing import Tuple
from numpy import ndarray


# 2.------------------- constants ------------------- #
attrs = ["AMB_TEMP", "CH4", "CO", "NMHC", "NO", "NO2",
         "NOx", "O3", "PM10",
         "PM2.5",  # <-- target (index = 9)
         "RAINFALL", "RH", "SO2", "THC", "WD_HR",
         "WIND_DIREC", "WIND_SPEED", "WS_HR"]


# 3.------------------- utilities ------------------- #
def get_N_hours_feat(
    month_data: ndarray,
    N: int,
    UNBIASED: bool = False,
  ) -> Tuple[ndarray, ndarray]:
    """ Get features of N hours. """

    #              month_data.shape = (#date,  18, 24)
    data  = month_data.transpose((0, 2, 1))  
                            # shape = (#date,  24, 18)
    data  = data.reshape(-1, 18)             
                            # shape = (#date * 24, 18)
    label = data[:, attrs.index("PM2.5")]
    total_hours = len(label)
    
    # set up an empty feature framework
    feats = (
        np.empty([0, N * 18 + 1])  # biased (default)
            if not UNBIASED else
        np.empty([0, N * 18])      # bias removed
    ) 
    # i represent one piece of data
    for i in range(total_hours - N):
        cur_feat = data[i : i + N].flatten()
        if not UNBIASED:           # Adding `w0` \
                                   #         (default)
            cur_feat = np.append(cur_feat, [1])
        cur_feat = np.expand_dims(cur_feat, axis=0)  # add new axis so to make it fit feats frame

        # aggregate together
        feats = np.concatenate([feats, cur_feat])

    label = label[N:] # use prev hour attr to predict current PM

    return feats, label


def read_train_csv(
    filename: str,
    N: int,
    UNBIASED: bool = False,
  ) -> Tuple[ndarray, ndarray]:
    """ A utility function for reading the training 
        data in CSV format. """

    data = (np.genfromtxt(
                filename,
                delimiter=',',  # CSV
                skip_header=1,  # skip the header
            )[:, 3:]            # omit station names
            .astype(float)      # read into numbers

            .reshape(12,        # 12 months,
                     -1,        # 20 days per month,
                   # ^^ -->       -1: filled \
                   #                     automatically
                     18,        # 18 features per day,
                     24))       # 24 hours per day

    # get January
    train_X, train_y = get_N_hours_feat(
                        data[0], N, UNBIASED=UNBIASED)

    for i in range(1, 12):  # Feb. ~ Dec.
        X, y = get_N_hours_feat(
                        data[i], N, UNBIASED=UNBIASED)
        train_X = np.concatenate((train_X, X), axis=0)
        train_y = np.concatenate((train_y, y), axis=0)

    return train_X, train_y


def read_valid_csv(
    filename: str,
    N: int,
    isval: bool = True,
    UNBIASED: bool = False,
  ) -> Tuple[ndarray, ndarray]:
    """
    A utility function for reading the testing data 
        in CSV format.
    `isval` is for validation.
    """
    DAYS = np.array([31, 28, 31, 30, 31, 30,
                     31, 31, 30, 31, 30, 31])

    if not isval:  # istest, hidden (deprecated)
        test_days = DAYS - 22
        cumul_days = [sum(test_days[:i]) 
                      for i in range(1, 12 + 1)]
    else:          # isval
        # 2 days for each month
        cumul_days = [2 * i for i in range(1, 12 + 1)]

    data = (np.genfromtxt(
                filename,
                delimiter=',',  # CSV
                skip_header=1,  # skip the header
            )[:, 3:]            # omit station names
            .astype(float)      # read into numbers

            .reshape(-1,        # -1: filled \
                                #        automatically
                     18,        # 18 features per day,
                     24))       # 24 hours per day


    test_X, test_y = get_N_hours_feat(
           data[:cumul_days[0]], N, UNBIASED=UNBIASED)

    for i in range(1, 12):
        X, y = get_N_hours_feat(
            data[cumul_days[i - 1] : cumul_days[i]], 
            N=N, UNBIASED=UNBIASED)
        test_X = np.concatenate((test_X, X), axis=0)
        test_y = np.concatenate((test_y, y), axis=0)

    return test_X, test_y


def read_test_csv(
    filename: str,
    N: int,
    UNBIASED: bool = False,
  ) -> Tuple[ndarray, ndarray]:
    """ A utility function for reading the testing 
        data in CSV format. """
    with open(filename) as f:
        parsed_N = int(f.readline().split(',')[0]
                                   .split('|')[-1])
        assert N == parsed_N, "N wrong!"

    data       = np.genfromtxt(
                     filename,
                     delimiter=',',  # CSV
                     dtype=str,       
                           # read into strings for ids
                     skip_header=1,  # skip the header
                 )
    id_columns = data[:, 0]          # testdata id
    test_X     = data[:, 1:].astype(float)  
                                     # read into \
                                     #     numbers
    if not UNBIASED:
        test_X = np.concatenate([
            test_X,
            np.ones((len(test_X), 1)),  # bias terms
        ], axis=1)

    return test_X, id_columns


# 4.------------------- main class ------------------- #
class LinearRegression(object):
    """
    A class wrapper for linear regression.

    Attributes
    ----------
    w : ndarray
        The weight vector for linear regression.
    C = [1, x1, x2, ... , xn]
    Pw = (C_T * C)^ -1 * C_T * y
    pred = C * Pw
    Methods
    -------
    train(X, y):
        Training the regressor by X and y.
    predict(X):
        Predict from X and the w vector.
    """
    def __init__(self) -> None: 
        self.w = np.array([])  

    def train(self, X: ndarray, y: ndarray) -> None:
        """ Input X and y to train the w vector. """
#         self.w = np.array([])  # TODO: Part 1 / 4
        a1 = X.T @ X
        a2 = np.linalg.inv(a1)
        a3 = a2 @ X.T
        self.w = a3 @ y

    def predict(self, X: ndarray) -> ndarray:
        """ Predict by the given X and trained w. """
#         pred_y = np.array([])  # TODO: Part 2 / 4
        pred_y = X @ self.w
        return pred_y


# 5.------------------- main function ------------------- #
def MSE(pred_y: ndarray, real_y: ndarray) -> float:
    """ Return the MSE by predicted and real data. """
    e = real_y - pred_y
    error = [i * i for i in e]
    error = sum(error)/len(error)
    return error
