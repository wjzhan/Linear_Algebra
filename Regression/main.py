#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 1. ------------------- libraries ------------------- #
import matplotlib.pyplot as plt  # plotting
from typing import Tuple         # for type hint
# local libraries (functions and classes from `lg.py`)
from lg import (
    read_train_csv, read_valid_csv, read_test_csv,
    LinearRegression, MSE)


# 2. ------------------- utilities ------------------- #
def run_train_valid(
    N: int,
    train_filename: str,
    valid_filename: str,
  ) -> Tuple[float, float]:
    """ Running the training and validation process.
        Return the two losses. """
    model = LinearRegression()

    # ------------------- train -------------------- #
    train_X, train_Y = read_train_csv(
                                    train_filename, N)

    model.train(train_X, train_Y)
    pred_train_Y = model.predict(train_X)
    train_loss = MSE(pred_train_Y, train_Y)
    # ------------------- valid -------------------- #
    valid_X, valid_Y = read_valid_csv(
           valid_filename, N, "val" in valid_filename)

    pred_valid_Y = model.predict(valid_X)
    valid_loss = MSE(pred_valid_Y, valid_Y)

    print(f"N = {N:2d} --> "
                   f"train_loss = {train_loss:8.4f}\t"
                   f"valid_loss = {valid_loss:8.4f}")
    return train_loss, valid_loss


def run_train_test(
    N: int,
    train_filename: str,
    test_filename: str,
    outcome_filename: str,
  ) -> Tuple[float, str]:
    """ Running the training and testing process. """
    model = LinearRegression()

    # ------------------- train -------------------- #
    train_X, train_y = read_train_csv(
                                    train_filename, N)

    model.train(train_X, train_y)
    pred_train_y = model.predict(train_X)
    train_loss = MSE(pred_train_y, train_y)
    # ------------------- test -------------------- #
    test_X, cols = read_test_csv(
             test_filename, N, "val" in test_filename)

    pred_test_y = model.predict(test_X)

    print(f"N = {N:2d} --> MSE = {train_loss}")
    print(f"N = {N:2d} --> testing...")
    with open(outcome_filename, 'w') as fout:
        fout.write(f"testdata_id|{N},"
                   f"PM2.5__DAY{N + 1:03d}\n")
        for _id, _data in zip(cols, pred_test_y):
            fout.write(f"{_id},{_data}\n")
    return train_loss, outcome_filename


def plot_N_loss(
    train_filename:   str = "data/train.csv",
    valid_filename:   str = "data/validation.csv",
    savefig_filename: str = "result.png",
  ) -> None:
    """ 
    Plot the loss after running `run_train_valid`.
    """
    all_train_losses = []
    all_valid_losses = []

    for N in range(1, 30 + 1):  # 1 ~ 30
        train_loss, valid_loss = run_train_valid(
                    N, train_filename, valid_filename)
        all_train_losses.append(train_loss)
        all_valid_losses.append(valid_loss)

    Ns = list(range(1, 30 + 1))  # 1 ~ 30
    plt.plot(Ns, all_train_losses, 
                              "b", label="train")
    plt.plot(Ns, all_valid_losses, 
                              "r", label="validation")
    plt.legend()
    plt.xlabel("N"); plt.ylabel("MSE loss")
    plt.savefig(savefig_filename)


# 3. ------------------- main ------------------- #
def main():
    plot_N_loss()

    N = 20
    try:
        assert N != 0, "`N` should be changed!"
    except AssertionError as err:
        print("\n`N` should not be 0...")
        N = int(input("Please choose an `N` "
                      "from above: "))
    test_filename = f"data/test_{N}.csv"
    while True:
        try:
            with open(test_filename) as f:
                pass
            break
        except FileNotFoundError as err:
            print(f"{test_filename} is required for "
                   "prediction!")
            import sys
            sys.exit(0)

    run_train_test(N, train_filename="data/train.csv",
                      test_filename=test_filename,
                      outcome_filename=(
                               f"prediction_{N}.csv"))


# 4. -------------------the entry point ------------------- #
if __name__ == "__main__": main()
