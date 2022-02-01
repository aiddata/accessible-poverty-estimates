# -*- coding: utf-8 -*-

"""Utility methods for Exploratory Data Analysis and Pre-processing"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

import seaborn as sns
from sklearn import preprocessing
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import percentileofscore
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

TM_pal_categorical_3 = ("#ef4631", "#10b9ce", "#ff9138")
sns.set(
    style="white",
    font_scale=1,
    palette=TM_pal_categorical_3,
)

SEED = 42
np.random.seed(SEED)

#### Scoring Helper Functions ####
def pearsonr2(estimator, X, y_true):
    """Calculates r-squared score using pearsonr

    Parameters
    ----------
    estimator
        The model or regressor to be evaluated
    X : pandas dataframe or a 2-D matrix
        The feature matrix
    y : list of pandas series
        The target vector

    Returns
    ----------
    float
        R2 using pearsonr
    """
    y_pred = estimator.predict(X)
    return pearsonr(y_true, y_pred)[0]**2


def mae(estimator, X, y_true):
    """Calculates mean absolute error

    Parameters
    ----------
    estimator
        The model or regressor to be evaluated
    X : pandas dataframe or a 2-D matrix
        The feature matrix
    y : list of pandas series
        The target vector

    Returns
    ----------
    float
        Mean absolute error
    """
    y_pred = estimator.predict(X)
    return mean_absolute_error(y_true, y_pred)


def rmse(estimator, X, y_true):
    """Calculates root mean squared error

    Parameters
    ----------
    estimator
        The model or regressor to be evaluated
    X : pandas dataframe or a 2-D matrix
        The feature matrix
    y : list of pandas series
        The target vector

    Returns
    ----------
    float
        Root mean squared error
    """
    y_pred = estimator.predict(X)
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2(estimator, X, y_true):
    """Calculates r-squared score using python's r2_score function

    Parameters
    ----------
    estimator
        The model or regressor to be evaluated
    X : pandas dataframe or a 2-D matrix
        The feature matrix
    y : list of pandas series
        The target vector

    Returns
    ----------
    float
        R-squared score using python's r2_score function
    """
    y_pred = estimator.predict(X)
    return r2_score(y_true, y_pred)


def mape(estimator, X, y_true):
    """Calculates mean average percentage error

    Parameters
    ----------
    estimator
        The model or regressor to be evaluated
    X : pandas dataframe or a 2-D matrix
        The feature matrix
    y : list of pandas series
        The target vector

    Returns
    ----------
    float
        Mean average percentage error
    """
    y_pred = estimator.predict(X)
    return np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100


def adj_r2(estimator, X, y_true):
    """Calculates adjusted r-squared score

    Parameters
    ----------
    estimator
        The model or regressor to be evaluated
    X : pandas dataframe or a 2-D matrix
        The feature matrix
    y : list of pandas series
        The target vector

    Returns
    ----------
    float
        Adjusted r-squared score
    """
    y_pred = estimator.predict(X)
    r2 = r2_score(y_true, y_pred)
    n = X.shape[0]
    k = X.shape[1]
    adj_r2 = 1 - (((1-r2)*(n-1))/(n - k - 1))

    return adj_r2


def percentile_ranking(series):
    """Converts list of numbers to percentile and ranking

    Parameters
    ----------
    series : pandas Series
        A series of numbers to be converted to percentile ranking

    Returns
    ----------
    list (of floats)
        A list of converted percentile values using scipy.stats percentileofscore()
    list (of ints)
        A list containing the ranks
    """
    percentiles = []
    for index, value in series.iteritems():
        curr_index = series.index.isin([index])
        percentile = percentileofscore(series[~curr_index], value)
        percentiles.append(percentile)
    ranks = series.rank(axis=0, ascending=False)

    return percentiles, ranks


#### Plotting Helper Functions ####

def plot_hist(data, title, x_label, y_label, bins=30):
    """Plots histogram for the given data

    Parameters
    ----------
    data : pandas Series
        The data to plot histogram
    title : str
        The title of the figure
    x_label : str
        Label of the x axis
    y_label : str
        Label of the y-axis
    bins : int
        Number of bins for histogram
    """
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_regplot(
    data,
    x_label='Wealth Index',
    y_label='Average Nightlight Intensity',
    y_var='ntl_mean'
):
    """Produces the regression plot for the given data

    Parameters
    ----------
    data : pandas Series
        The data to plot regression plot
    x_var : str
        The variable name of the x-axis
    y_var : str
        The variable name of the y-axis
    x_label : str
        Label of the x axis
    y_label : str
        Label of the y-axis
    """
    ax = sns.regplot(
        x=x_label,
        y=y_var,
        data=data,
        lowess=True,
        line_kws={"color": "black", "lw": 2},
        scatter_kws={"alpha": 0.3},
    )
    plt.ticklabel_format(style='sci', axis='x', scilimits=(1,5))
    plt.title(
        "Relationship between {} \nand {}".format(
            x_label, y_label
        )
        + r" ($\rho$ = %.2f, $r$ =%.2f)"
        % (
            spearmanr(
                data[x_label].tolist(), data[y_var].tolist()
            )[0],
            pearsonr(
                data[x_label].tolist(), data[y_var].tolist()
            )[0],
        )
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_corr(
    data,
    features_cols,
    indicator="Wealth Index",
    figsize=(5, 6),
    max_n=30,
    output_file=None
):
    """Produces a barplot of the Spearman rank correlation and Pearson's correlation
    for a group of values in descending order

    Parameters
    ----------
    data : pandas DataFrame
        The dataframe containing the feature columns
    feature_cols : str
        The list of feature column names in the data
    indicator : str (default is "Wealth Index")
        The socioeconomic indicator to correlate each variable with
    figsize : tuple (default is (5,6))
        Size of the figure
    max_n : int
        Maximum number of variables to plot
    output_file : str
        The desired pathway to output the plot
    """
    n = len(features_cols)
    spearman = []
    pearsons = []
    for feature in features_cols:
        spearman.append(
            ( feature, spearmanr(data[feature], data[indicator])[0] )
        )
        pearsons.append(
            ( feature, pearsonr(data[feature], data[indicator])[0] )
        )
    spearman = sorted(spearman, key=lambda x: abs(x[1]))
    pearsons = sorted(pearsons, key=lambda x: abs(x[1]))
    #
    plt.figure(figsize=figsize)
    plt.title( "Spearman Correlation Coefficient for {}".format(indicator) )
    plt.barh(
        [x[0] for x in spearman[n - max_n :]],
        [x[1] for x in spearman[n - max_n :]],
    )
    plt.grid()
    #
    plt.figure(figsize=figsize)
    plt.title( "Pearsons Correlation Coefficient for {}".format(indicator) )
    plt.barh(
        [x[0] for x in pearsons[n - max_n :]],
        [x[1] for x in pearsons[n - max_n :]],
    )
    plt.grid()
    if output_file:
        plt.savefig(fname=output_file)
    plt.show(block=False)



#### Data subsetting for model efficiency

def corr_finder(X, threshold):
    """ For each variable, find the independent variables that are equal to
        or more highly correlated than the threshold with the curraent variable

    Parameters
    ----------
    X : pandas Dataframe
        Contains only independent variables and desired index
    threshold: float < 1
        Minimum level of correlation to search for

    Returns
    -------
    Dictionary with the key's as independent variavble indices and values as a
    list of variables with a correlation greater to or equal than the threshold.

    Correlation Matrix
    """

    corr_matrix = X.corr(method='kendall') #create the correlation matrix
    corr_dic = {}
    for row_name, ser in corr_matrix.iterrows(): #search through each row
        corr_list = [] #list of variables past/at the threshold
        for idx, val in ser.iteritems():  #search through the materials of each row
            if (abs(val) > threshold) and (abs(val) != 1): #if the variable correlates past/at the threshold
                corr_list.append(idx)
        corr_dic[row_name] = corr_list
    return corr_dic, corr_matrix


def subset_dataframe(df,feature_cols, remove_cols):
    """ Create and return a copy of the desired dataframe without the listed columns.

    Parameters
    ----------
    df : pandas Dataframe
        Dataframe from which subset will occur

    feature_cols: list of strings
        Column labels of features to be used in analysis

    remove_cols: list of strings
        Column labels for the columns to be removed

    Return
    ------
    Subsetted pandas DataFrame and a new feature columns list
    """

    updated_cols = [col for col in df.columns if col not in remove_cols ]  #update list of column names for dataframe
    new_features = [col for col in feature_cols if col not in remove_cols] #update column list of feature columns
    return df.copy()[updated_cols], new_features

