# -*- coding: utf-8 -*-

"""Utility methods for Exploratory Data Analysis and Pre-processing"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from math import log, ceil
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
import re

import seaborn as sns
from sklearn import preprocessing
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import percentileofscore
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error



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

def mape(estimator, X, y_true):
    """Calculates mean absolute percentage error

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
        Mean absolute percentage error
    """
    y_pred = estimator.predict(X)
    return mean_absolute_percentage_error(y_true, y_pred)

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

def plot_hist(
    data,
    title,
    x_label,
    y_label,
    bins=30,
    output_file=None,
    show=False
):
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
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if output_file:
        plt.savefig(fname=output_file, bbox_inches="tight")
    if show:
        plt.show(block=False)


def plot_regplot(
    data,
    x_label='Wealth Index',
    y_label='Average Nightlight Intensity',
    y_var='ntl_mean',
    output_file=None,
    show=False
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
    plt.figure()
    ax = sns.regplot(
        x=x_label,
        y=y_var,
        data=data,
        lowess=True,
        line_kws={"color": "black", "lw": 2},
        scatter_kws={"alpha": 0.3},
    )
    plt.ticklabel_format(style='sci', axis='x', scilimits=(1,5))
    r2 = pearsonr(data[x_label].tolist(), data[y_var].tolist())[0]
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
    if output_file:
        plt.savefig(fname=output_file, bbox_inches="tight")
    if show:
        plt.show(block=False)
    return r2


def plot_corr(
    data,
    features_cols,
    indicator="Wealth Index",
    method="pearsons",
    figsize=(5, 6),
    max_n=30,
    output_file=None,
    show=False
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
    if method == 'pearsons':
        name = 'Pearsons'
        func = pearsonr
    elif method == 'spearman':
        name = 'Spearman'
        func = spearmanr
    else:
        raise ValueError(f'Invalid method provided ({method})')

    values = []
    for feature in features_cols:
        values.append(
            ( feature, func(data[feature], data[indicator])[0] )
        )

    values = sorted(values, key=lambda x: abs(x[1]))

    plt.figure(figsize=figsize)
    plt.title( f"{name} Correlation Coefficient for {indicator}" )
    plt.barh(
        [x[0] for x in values[n - max_n :]],
        [x[1] for x in values[n - max_n :]],
    )
    plt.grid()
    if output_file:
        plt.savefig(fname=output_file, bbox_inches="tight")
    if show:
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

#import plotly.graphics_objects as go
#import re
#Also, add plotly to list of requirements
def plot_parallel_coordinates(
    cv_results,
    output_name = 'Unidentified Region',
    show = False,
    output_file = None,
    color_scale = None,
    show_colorbar = True,
    logistic_params=dict(),
    visual_mode = "dark"):

    """Produces a parallel coordinates plot that displays the relationship between the various
    hyperparameter configurations used during grid search and the corresponding test scores.

    Parameters
    ----------
    cv_results : expects a cv_results_ dictionary, an attribute of a fitted GridSearchCV object
        The dictionary containing the data to analyze (the mean test scores of every
        hyperparameter configuration executed during a grid search)
    output_name : str (default is "Unidentified Region")
        The country in question, used for titling the graph. Currently uses the
        corresponding "output_name" from config.ini
        (e.g., BJ_2017-18_DHS for benin)
    show : bool (default is False)
        Whether to display parallel coordinates plot on screen or not
    output_file : str (default is None)
        The desired pathway to output the plot. If set to None, no file is saved.
    color_scale : str or list (default is None)
        The desired color scale to use for indicating score. Options can be found in the
        documentation for colorscales for Plotly parcoords lines, currently located at
        https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.parcoords.html#plotly.graph_objects.parcoords.Line.colorscale
        If set to None, color_scale is automatically set according to the selected visual mode
    show_colorbar : bool (default is True)
        Whether to display the separate colorbar that indicates how scores are colorcoded
    logistic_params : dictionary whose keys are strings and whose values are integers
    (default is empty dictionary)
        Dictionary whose keys are the shortened names of the quantitative hyperparameters
        whose axes to display on a logistic scale rather than a linear scale. Corresponding
        values are the desired integer base to use for logistic scaling. If set to empty,
        all quantitative hyperparameters will be displayed on a linear scale.
            -e.g., {"n_estimators": 10, "max_depth": 2} for approximate values of
            10, 100, and 1000 for n_estimators and approximate values of 3, 6, and 12
            for max_depth.
    visual_mode : str (default is "dark")
        The overall color theme of the plot. Currently two options: "dark" and "light".
    """
    color_dict = {'dark': 'picnic_r', 'light': 'RdBu'}
    if color_scale is None:
        color_scale = color_dict[visual_mode]

    if 'mean_test_score' not in cv_results.keys():
        cv_results['mean_test_score'] = cv_results.pop('mean_test_r2')                  #Set scoring metric to score of choice from scoring dict

    df = pd.DataFrame(cv_results)
    # print("Default: \n", df, type(df))

    category_dict=dict()  #store dict of lists, where each key/value pair represents a categorical hyperparameter and its list of mappings between categorical and quantitative values
    for column in df.columns:
        if ((not re.search('mean_test_score|param_', column)) or (column != 'mean_test_score' and len(df[column].unique()) == 1)): # Remove columns from dataframe that don't contain parameters with variation or the score under consideration
            df.drop(column, inplace = True, axis = 1)
        else: #if the column is valid, transform the data into valid int types and shorten name of parameters
            if column == 'mean_test_score':
                df[column] = df.pop(column)
            else:
                shortened = column.rsplit("param_regressor__", 1)[1] #Remove long "param_regressor__" phrase from parameters
                df[shortened] = df[column]
                df.drop(column, inplace = True, axis = 1)
                try:
                    df[shortened] = pd.to_numeric(df[shortened])
                except: #Convert categorical data to quantitative data if necessary, storing mappings in category_list
                    category_dict[shortened] = df[shortened].unique().tolist()
                    df[shortened] = df[shortened].apply(lambda x: category_dict[shortened].index(x))



    # print("After filter: \n", df, type(df))
    # # df.drop(labels = ["param_regressor__criterion", "param_regressor__bootstrap"], inplace=True, axis = 1)
    # # df = df.filter(regex='mean_test_score|param_')
    # # df.drop(["param_regressor__criterion", "param_regressor__bootstrap"], inplace=True, axis = 1)
    # print(df.dtypes)

    # print("Shortened df: \n", df, type(df))
    # print(df.columns)
    # print(df.dtypes)
    # print(category_dict)


    # for col in df.columns:
    #     print(col)
    #     print(df[col].values[3], "type: ", type(df[col].values[3]))
    #     print(df[col])

    col_list = []

    for col in df.columns:
        if col in category_dict:
            col_dict = dict(
                label=col,
                tickvals=list(range(len(category_dict[col]))),
                ticktext=category_dict[col],
                values = df[col]
            )
        else:
            if col[:4] == "min_": #Invert columns for "min_" hyperparams so values with typically higher scores (lower values)
                                                # are placed next to high scores more often, minimizing crossing lines in graph
                dim_range = [df[col].max(), df[col].min()]
            else:
                dim_range = [df[col].min(), df[col].max()]
            if col in logistic_params:
                logged_vals = df[col].apply(lambda x: log(x, logistic_params[col]))
                tickvals = np.unique(np.array(list(range(ceil(max(logged_vals))+1)) + list(logged_vals)))
                col_dict = dict(
                    range = dim_range,
                    label = col,
                    values = logged_vals,
                    tickvals = tickvals,
                    ticktext = list(round(pow(logistic_params[col], x)) for x in tickvals)
                )
            else:
                col_dict = dict(
                    range = dim_range,
                    label = col,
                    values=df[col]
                )
        col_list.append(col_dict)

    parCoords = go.Parcoords(
        dimensions=col_list,
        line=dict(color=df['mean_test_score'], colorscale=color_scale)
    )

    fig = go.Figure(
        data = parCoords
    )

    if visual_mode == "dark":  #Display Theming
        fig.update_layout(
            paper_bgcolor='#000',
            plot_bgcolor='#000',
            title_text='Parallel Coordinates Plot for ' + output_name,
            font_color='#DDD',
            font_size=18
        )
    else:
        fig.update_layout(
            paper_bgcolor='#FFF',
            plot_bgcolor='#FFF',
            title_text='Parallel Coordinates Plot for ' + output_name,
            font_color='#000',
            font_size=18
        )

    fig.update_traces(
        line_showscale=show_colorbar, #Show colorbar
        selector=dict(type='parcoords')
    )

    if output_file:
        fig.write_html(output_file + '.html')
        fig.write_image(output_file + '.png')
    if show:
        fig.show()

