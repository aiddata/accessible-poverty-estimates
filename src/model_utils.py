# -*- coding: utf-8 -*-

"""Utility methods for evaluating Wealth Prediction Models"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import random
import seaborn as sns
from scipy import stats
from math import sqrt
from joblib import dump, load

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import xgboost as xgb
import mlflow

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    cross_val_predict,
)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    PolynomialFeatures,
)

from sklearn.svm import SVR, SVC
from sklearn.inspection import permutation_importance
import seaborn as sns


TM_pal_categorical_3 = ("#ef4631", "#10b9ce", "#ff9138")
sns.set(
    style="white",
    font_scale=1.25,
    palette=TM_pal_categorical_3,
)

SEED = 42
np.random.seed(SEED)

def evaluate_model(
    data,
    feature_cols,
    indicator_cols,
    clust_str,
    model_name=None,
    wandb=None,
    model_type="ridge",
    scoring={"r2": "r2"},
    refit="r2",
    search_type="random",
    n_splits=5,
    n_iter=50,
    task_type="regression",
    plot_importance=False,
    figsize=(5, 8),
    std_scale=False,
    minmax_scale=False,
    polynomial=False,
    poly_degree=2,
    n_workers=-1,
    verbose=1,
    plot=True,
    output_file=None,
    show=False
):
    """ Automatically trains and evaluates the specified model on given dataset
    using an n-fold nested cross validation scheme. Supported models so far are:
    ridge regression, random forest regression, and xgboost regression.

    Parameters
    ----------
    data : pandas DataFrame
        A pandas dataframe containing the feature and indicator columns
    feature_cols : list
        The list of predictive features to be used for training
    indicator_cols : list
        The list of socioeconomic indicators to be predicted
    model_type : str (default is 'ridge')
        Type of model to use. Supported types so far are: 'ridge', 'random_forest', and 'xgboost'
    scoring : dict (default is {"r2": "r2"})
        A dictionary containing the scoring metrics to be used
    refit : str (default is 'r2')
        Scoring metric to be optimized
    search_type : str (default is random)
        Search type: either 'grid' or 'random'
    n_splits : int (default is 5)
        Number of splits/folds for the n-fold cross validation
    n_iter : int (default is 50)
        Number of iterations for the random search cross validation
    task_type : str (default is 'regression')
        The type of task: either 'classification' or 'regression'
    plot_importance : bool (default is False)
        Indicates whether or not to plot feature importance. Applicable to random forest or xgboost only.
    figsize : tuple (default is (5,8))
        Size of feature importance plot
    std_scale : bool (default is False)
        Indicates whether or not to apply standard scaling to features
    minmax_scale : bool (default is False)
        Indicates whether or not to apply min-max scaling to features

    Returns
    ----------
    pandas DataFrame
        Contains cluster number, actual, and predicted socioeconomic indicators
    """
    print("Evaluating model for: ", output_file)

    # Set up parameter grid
    param_grid = get_param_grid(model_type)

    # Initialize results dictionary
    results = {
        indicator + type_: []
        for indicator in indicator_cols
        for type_ in ["_pred", "_true"]
    }

    # Iterate over the socioeconomic indicators
    for index, indicator in enumerate(indicator_cols):
        X = data[feature_cols]
        y = data[indicator].tolist()
        clusters = data[clust_str].tolist()

        # Instantiate model
        model = get_model(model_type)


        # Nested cross validation
        cv, nested_scores, y_true = nested_cross_validation(
            model,
            X,
            y,
            param_grid=param_grid,
            scoring=scoring,
            refit=refit,
            search_type=search_type,
            n_splits=n_splits,
            n_iter=n_iter,
            task_type=task_type,
            std_scale=std_scale,
            minmax_scale=minmax_scale,
            polynomial=polynomial,
            poly_degree=poly_degree,
            n_workers=n_workers,
            verbose=verbose,
        )
        # print("RESULTS 1B: ", cv.cv_results_)


        # Display scores
        print(
            "Socioeconomic indicator: {}".format(indicator)
        )
        for score in nested_scores:
            print(
                score,
                ": %.4f" % nested_scores[score].mean(),
            )
            # mlflow.log_param(score, nested_scores[score].mean())
            # mlflow.log_param(f'{score}_list', nested_scores[score])
            mlflow.log_metric(f"mean_{score}", nested_scores[score].mean())

            print(nested_scores[score])
            if score == 'test_r2':
                r_squared = nested_scores[score].mean()

        # formatted_indicator = ' '.join([x for x in indicator.split() if '(' not in x]).title()
        # if wandb is not None:
        #     wandb.log({'{} R-squared'.format(formatted_indicator): r_squared})

        # Plot results
        # if plot:
        #     plot_cross_val_results(
        #         y_true,
        #         y_pred,
        #         formatted_indicator,
        #         nested_scores,
        #         wandb=wandb,
        #         refit=refit,
        #         output_file=output_file  + f"cross_val_{index}.png",
        #         show=show
        #     )

        print("Fitting CV on full dataset")
        # Get best estimator
        print("About to fit: ")
        print(type(cv))
        cv.fit(X, y)

        # Log each feature's importance as a MLflow metric
        for z in range(len(X.columns)):
            mlflow.log_metric(f"{X.columns[z]}_importance",
                              cv.best_estimator_.named_steps["regressor"].feature_importances_[z])



        # if mlflow.active_run() is not None:
        #     mlflow.sklearn.log_model(cv.best_estimator_,
        #                              "best model",
        #                              registered_model_name=model_name)
        #     for k in cv.best_params_.keys():
        #         mlflow.log_param(k, cv.best_params_[k])


        print(
            "Best estimator: {}".format(cv.best_estimator_)
        )

        # Save results
        # results[indicator + "_pred"] = y_pred
        results[indicator + "_true"] = y_true
        results[clust_str] = clusters

        # Plot feature importances (for tree-based models only)
        if plot_importance:
            if model_type == "random_forest":
                rf_feature_importance(
                    cv, X, y, size=figsize,
                    output_file=output_file + f"rf_feature_importance_{index}.png",
                    show=show
                )
                mlflow.log_artifact(output_file + f"rf_feature_importance_{index}.png")

                rf_permutation_importance(
                    cv, X, y, size=figsize,
                    output_file=output_file + f"rf_permutation_importance_{index}.png",
                    show=show
                )
                mlflow.log_artifact(output_file + f"rf_permutation_importance_{index}.png")


            elif model_type == "xgboost":
                xgb_feature_importance(
                    cv, X, y, size=figsize,
                    output_file=output_file + f"xgb_feature_importance_{index}.png",
                    show=show
                )
                mlflow.log_artifact(output_file + f"xgb_feature_importance_{index}.png")

    return cv #, pd.DataFrame(results)


def get_param_grid(model_type='ridge'):
    """Returns the model parameter grid to be used as input for cross validation
    hyper parameter optimization

    Parameters
    ----------
    model_type : str (default is 'ridge')
        Type of model to use. Supported types so far are: 'ridge', 'random_forest',
        and 'xgboost'

    Returns
    ----------
    dict
        A dictionary of parameters
    """
    np.random.seed(SEED)
    if (model_type == "ridge") or (model_type == "lasso"):
        param_grid = {
            "regressor__alpha": stats.uniform.rvs(loc=0, scale=4, size=3),
            "regressor__normalize": [True, False],
        }
    if model_type == "elastic_net":
        param_grid = {
            "regressor__alpha": stats.uniform.rvs(loc=0, scale=4, size=3),
            "regressor__l1_ratio": np.random.uniform(0, 1, 10),
            "regressor__normalize": [True, False],
        }
    #the following random forest hyperparameters are explained in the paper
    elif model_type == "random_forest":
        param_grid = {
            # "regressor__n_estimators": stats.randint(200, 2000),
            # "regressor__max_features": ["auto", "sqrt", "log2"],
            # "regressor__max_depth": stats.randint(3, 10),
            # "regressor__min_samples_split": stats.randint(2, 10),
            # "regressor__min_samples_leaf": stats.randint(1, 10),
            # "regressor__bootstrap": [True, False],

            # "regressor__criterion": ["squared_error"],
            # "regressor__n_estimators": [100, 500, 1000, 2000],
            # "regressor__max_features": [0.33, "sqrt"],
            # "regressor__max_depth": [5, 10, 25],
            # "regressor__min_samples_split": [2, 8, 16],
            # "regressor__min_samples_leaf": [1, 4, 8],
            # "regressor__bootstrap": [True]

            # "regressor__criterion": ["squared_error"],
            # "regressor__n_estimators": [500, 1000],
            # "regressor__max_features": [0.33],
            # "regressor__max_depth": [10, 25],
            # "regressor__min_samples_split": [2, 4],
            # "regressor__min_samples_leaf": [1, 2],
            # "regressor__bootstrap": [True]

            "regressor__criterion": ["squared_error"],
            "regressor__n_estimators": [500],
            "regressor__max_features": [0.33],
            "regressor__max_depth": [10],
            "regressor__min_samples_split": [2],
            "regressor__min_samples_leaf": [1],
            "regressor__bootstrap": [True]
        }
    elif model_type == "xgboost":
        param_grid = {
            "regressor__n_estimators": stats.randint(200, 2000),
            "regressor__learning_rate": np.random.uniform(1e-3, 0.2, 100),
            "regressor__subsample": np.random.uniform(0.9, 1, 100),
            "regressor__max_depth": stats.randint(3, 10),
            "regressor__colsample_bytree": np.random.uniform(0.7, 1, 100),
            "regressor__min_child_weight": stats.randint(1, 5),
            "regressor__gamma": np.random.uniform(0.5, 5, 100),
        }
    elif model_type == "svr":
        param_grid = {
            "regressor__kernel": ["linear", "poly", "rbf"],
            "regressor__degree": stats.randint(1, 5),
            "regressor__gamma": ["auto", "scale"],
            "regressor__C": np.random.uniform(0, 10, 100),
        }
    return param_grid


def get_model(model_type='ridge'):
    """Returns the model instance to be used as input for cross validation

    Parameters
    ----------
    model_type : str (default is 'ridge')
        Type of model to use. Supported types so far are: 'ridge', 'random_forest',
        and 'xgboost'

    Returns
    ----------
    model instance
        Model to be evaluated
    """
    np.random.seed(SEED)
    if model_type == "ridge":
        model = Ridge(random_state=SEED)
    elif model_type == "lasso":
        model = Lasso(random_state=SEED)
    elif model_type == "elastic_net":
        model = ElasticNet(random_state=SEED)
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            random_state=SEED,
            n_jobs=-1
        )
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(
            objective="reg:linear",
            random_state=SEED,
            n_jobs=-1,
        )
    elif model_type == "svr":
        model = SVR()

    return model


def nested_cross_validation(
    model,
    X,
    y,
    param_grid,
    scoring,
    refit,
    search_type="random",
    n_splits=5,
    n_iter=50,
    std_scale=False,
    minmax_scale=False,
    polynomial=False,
    poly_degree=2,
    task_type="regression",
    n_workers=-1,
    verbose=0
):
    """An implementation of n-fold nested cross validation.

    Parameters
    ----------
    model:
        The model to be used for cross validation
    X : pandas DataFrame, numpy array, or 2D list
        Contains the feature matrix for training
    y : pandas Series or list
        Contains the target vector to predict
    param_grid : dict
        Contains the dictionary of parameters to be optimized.
        Keys are the parameter names, values contain the range of values for that parameter.
    scoring : dict (default is {"r2": "r2"})
        A dictionary containing the scoring metrics to be used
    refit : str (default is 'r2')
        Scoring metric to be optimized
    search_type : str (default is random)
        Search type: either 'grid' or 'random'
    n_splits : int (default is 5)
        Number of splits/folds for the n-fold cross validation
    n_iter : int (default is 30)
        Number of iterations for the random search cross validation
    task_type : str (default is 'regression')
        The type of task: either 'classification' or 'regression'
    std_scale : bool (default is False)
        Indicates whether or not to apply standard scaling to features
    minmax_scale : bool (default is False)
        Indicates whether or not to apply min-max scaling to features

    Returns
    ----------
    CV instance
        Either an instance of GridSearchCV or RandomSearchCV
    dict
        A dictionary containing the scores specified in the scoring dictionary
    """
    np.random.seed(SEED)

    # Define inner and outer cross validation folds
    if task_type == "classification":
        inner_cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=SEED,
        )
        outer_cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=SEED,
        )
    elif task_type == "regression":
        inner_cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=SEED,
        )
        outer_cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=SEED,
        )

    # Define pipeline: transformation + models
    pipeline = []
    if std_scale:
        std_scaler = StandardScaler()
        pipeline.append(("std_scale", std_scaler))
    if minmax_scale:
        minmax_scaler = MinMaxScaler()
        pipeline.append(("minmax_scale", minmax_scaler))
    if polynomial:
        poly = PolynomialFeatures(degree=poly_degree)
        pipeline.append(("poly", poly))
    if task_type == "regression":
        pipeline.append(("regressor", model))
    pipe = Pipeline(pipeline)

    # Define search type: grid or random
    if search_type == "grid":
        cv = GridSearchCV(
            estimator=pipe,
            scoring=scoring,
            param_grid=param_grid,
            cv=inner_cv,
            verbose=verbose,
            n_jobs=n_workers,
            refit=refit,
        )
    elif search_type == "random":
        cv = RandomizedSearchCV(
            estimator=pipe,
            n_iter=n_iter,
            scoring=scoring,
            param_distributions=param_grid,
            cv=inner_cv,
            verbose=verbose,
            random_state=SEED,
            n_jobs=n_workers,
            refit=refit,
        )
    else:
        raise ValueError("Unknown search type!")

    print("Running cross_validate")
    # Commence cross validation
    nested_scores = cross_validate(
        cv,
        X=X,
        y=y,
        cv=outer_cv,
        n_jobs=n_workers,
        scoring=scoring,
        verbose=verbose,
        return_train_score=True,
    )

    # Get cross validated predictions
    # y_pred = cross_val_predict(
    #     cv,
    #     X=X,
    #     y=y,
    #     cv=outer_cv,
    #     n_jobs=n_workers,
    #     verbose=verbose
    # )

    return cv, nested_scores, y


def plot_cross_val_results(
    y_true,
    y_pred,
    indicator,
    nested_scores,
    wandb,
    refit='r2',
    output_file=None,
    show=False
):
    """Plots cross validated estimates.

    Parameters
    ----------
    y_true : pandas Series or list
        Contains the ground-truth target vector
    y_pred : pandas Series or list
        Contains the predictions
    indicator : str
        A string value specifying the indicator
    nested_scores : dict
        A dictionary of output scores produced through cross validation
    refit : str (default is 'r2')
        Scoring metric to be optimized
    """

    # Get cross validation results
    #y_pred = cross_val_predict(cv_model, X, y, cv=n_splits)
    plt.figure()

    # Plot Actual vs Predicted
    ax = sns.regplot(
        y_true,
        y_pred,
        line_kws={"color": "black", "lw": 1},
        scatter_kws={"alpha": 0.3},
    )
    plt.title(
        indicator
        + r" $r^2: {0:.3f}$".format(
            nested_scores["test_" + str(refit)].mean()
        )
    )
    plt.xlabel("Observed " + indicator.lower())
    plt.xticks(rotation=30)
    plt.ylabel("Predicted " + indicator.lower())
    if wandb is not None:
        wandb.log({'{}'.format(indicator): wandb.Image(plt)})
    if output_file:
        plt.savefig(fname=output_file, bbox_inches="tight")
    print("show graph?")
    if show:
        print("show graph now!")
        plt.show(block=False)


def rf_feature_importance(
    cv, X, y, n_features=20, size=(10, 15),
    output_file=None,
    show=False
):
    """ Plots the feature importances for random forest regressor.

    Parameters
    ----------
    cv :
    X : pandas DataFrame, numpy array, or 2D list
        Contains the feature matrix for training
    y : pandas Series or list
        Contains the target vector to predict
    n_features : int
        Number of features to plot
    size : tuple
        Size of the figure
    """

    feat_impt = {}
    for z in range(len(X.columns)):
        feat_impt[
            X.columns[z]
        ] = cv.best_estimator_.named_steps[
            "regressor"
        ].feature_importances_[
            z
        ]
    plt.figure()

    pd.DataFrame(
        {"Feature Importance": feat_impt}
    ).sort_values(
        by="Feature Importance", ascending=False
    ).iloc[
        :n_features
    ].plot(
        kind="barh", figsize=size
    )
    plt.grid()
    plt.gca().invert_yaxis()
    if output_file:
        plt.savefig(fname=output_file, bbox_inches="tight")
    if show:
        plt.show(block=False)


def rf_permutation_importance(
    cv, X, y, n_features=20, size=(10, 15),
    output_file=None,
    show=False
):
    """ Plots the feature importances for random forest regressor.

    Parameters
    ----------
    cv :
    X : pandas DataFrame, numpy array, or 2D list
        Contains the feature matrix for training
    y : pandas Series or list
        Contains the target vector to predict
    n_features : int
        Number of features to plot
    size : tuple
        Size of the figure
    """
    model = cv.best_estimator_.named_steps["regressor"]
    results = permutation_importance(model, X, y, n_repeats=10)
    raw_scores = results.importances
    median_score = np.median(raw_scores, axis=1)
    df = pd.DataFrame(index=X.columns, data=median_score, columns=["Permutation Importance"])

    plt.figure()

    df.sort_values(
        by="Permutation Importance", ascending=False
    ).iloc[
        :n_features
    ].plot(
        kind="barh", figsize=size
    )
    plt.grid()
    plt.gca().invert_yaxis()
    if output_file:
        plt.savefig(fname=output_file, bbox_inches="tight")
    if show:
        plt.show(block=False)


def xgb_feature_importance(
    cv, X, y, n_features=30, size=(10, 15)
):
    """ Plots the feature importances for XGBoost regressor.

    Parameters
    ----------
    cv :
    X : pandas DataFrame, numpy array, or 2D list
        Contains the feature matrix for training
    y : pandas Series or list
        Contains the target vector to predict
    n_features : int
        Number of features to plot
    size : tuple
        Size of the figure
    """
    plt.figure()

    fig, ax = plt.subplots(1, 1, figsize=size)
    xgb.plot_importance(
        cv.best_estimator_.named_steps["regressor"], ax=ax
    )
    plt.show()


def rf_feature_importance_dataframe(cv, X, y):
    """ Returns the feature importances for random forest regressor.

    Parameters
    ----------
    cv :
    X : pandas DataFrame, numpy array, or 2D list
        Contains the feature matrix for training
    y : pandas Series or list
        Contains the target vector to predict
    """
    feat_impt = {}
    for z in range(len(X.columns)):
        feat_impt[X.columns[z]] = cv.best_estimator_.named_steps["regressor"].feature_importances_[z]
    df = pd.DataFrame(
        {"feature_importance": feat_impt}
    ).sort_values(
        by="feature_importance", ascending=False
    )
    return df


def rf_permutation_importance_dataframe(cv, X, y):
    """ Returns the permutation importances for random forest regressor.

    Parameters
    ----------
    cv :
    X : pandas DataFrame, numpy array, or 2D list
        Contains the feature matrix for training
    y : pandas Series or list
        Contains the target vector to predict
    """
    results = permutation_importance(cv.best_estimator_.named_steps, X, y)
    raw_scores = results.importances
    df = pd.DataFrame(index=X.columns, data=raw_scores, columns='Raw Permutation Importance')
    return df


def save_model(cv, df, features, indicator, model_path):
    # define X,y for all data
    X = df[features]
    y = df[indicator].values.ravel()

    # refit cv model with all data
    best = cv.best_estimator_.fit(X, y)

    # save model
    dump(best, model_path)
