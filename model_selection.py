import csv
import numpy as np
from math import log
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

def parse_data(path):
    X, Y = [], [] 
    with open(path, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=' ', quotechar='|'))
        for row in data[1:]:
            line = row[0]
            line = line.split(',')[1:]
            line = [float(e) for e in line]
            X.append(line[:-1])
            Y.append(line[-1])
    return np.array(X), np.array(Y)
 
def calculate_bic(n, mse, num_params):
    bic = n * log(mse) + num_params * log(n)
    return bic

def model_selection(degrees, plot):
    X, Y = parse_data("./data.csv")
    mses = []
    bics = []
    params = []
    for degree in degrees:
        polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
        polyreg.fit(X, Y)
        num_params = degree * X.shape[1] + 1 
        yhat = polyreg.predict(X)
        mse = mean_squared_error(Y, yhat)
        bic = calculate_bic(len(Y), mse, num_params)
        mses.append(mse)
        bics.append(bic)
        params.append(num_params)
        print('Number of parameters: %d' % (num_params))
        print('MSE: %.3f' % mse)
        print('BIC: %.3f' % bic)
    if plot:
        plt.rcParams['axes.grid'] = True 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.2)
        ax1.plot(params, mses, 'tab:green')
        ax2.plot(params, bics)
        ax1.set(xlabel='Number of params', ylabel='Mean Squared Error')
        ax2.set(xlabel='Number of params', ylabel='BIC')
        fig.savefig("plot.png")

if __name__ == "__main__":
    degrees = range(1, 14)
    model_selection(degrees, True)









