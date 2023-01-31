import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import logging

from src.prep import bin_data


class Trainer:
    """Class for training a model with nested CV."""

    def __init__(self, logger=None):
        """Initialize class."""
        if logger is None:
            self.logger = logging.getLogger()
            self.logger.addHandler(logging.StreamHandler())
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

    def train(
        self, data, strat_col, label_col, pipeline, cv_inner, cv_outer, param_grid
    ):
        """Train model using nested cross-validation.
        Args:
            data (pd.DataFrame): dataframe to train on
            pipeline (sklearn.pipeline.Pipeline): pipeline to train
            strat_col (str): name of column to stratify on
            label_col (str): name of label column
            cv_inner (sklearn.model_selection._split): inner cross-validation
            cv_outer (sklearn.model_selection._split): outer cross-validation
            param_grid (dict): parameter grid for grid search
        Returns:
            y_pred (np.array): predicted values
            y_true (np.array): true values
            y_pred_std (np.array): standard deviation of predicted values
            y_pyment (np.array): pyment values
        """
        # report parameters
        self._print("Parameters:")
        self._print("  strat_col: {}".format(strat_col))
        self._print("  label_col: {}".format(label_col))
        self._print("  cv_inner: {}".format(cv_inner))
        self._print("  cv_outer: {}".format(cv_outer))
        self._print("  param_grid: {}".format(param_grid))
        self._print("  pipeline: {}".format(pipeline))

        # initialize arrays to store results
        y_pred = np.array([])
        y_true = np.array([])
        y_pred_std = np.array([])
        y_index = np.array([])
        y_fold = np.array([])

        # store coefficients in array if available
        coef = np.array([])

        # store best params in array
        best_params = np.array([])

        # index for outer CV !! make it pet_id
        index = np.arange(data.shape[0])

        # k-fold iterator for outer CV
        k = 1

        # store scores per fold
        scores = []

        # loop over outer folds
        for train_idx, test_idx in cv_outer.split(data, data[strat_col]):
            # get train and test data and index
            X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
            y_train, y_test = (
                data[label_col].iloc[train_idx],
                data[label_col].iloc[test_idx],
            )

            y_index_train, y_index_test = X_train.index.values, X_test.index.values

            # initiate grid search (inner CV)
            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv_inner,
                scoring="neg_mean_absolute_error",
                n_jobs=5,
            )

            # fit grid search
            grid.fit(X_train, y_train)
            
            # get best model
            best_pipe = grid.best_estimator_

            # fit best model on outer fold
            best_pipe.fit(X_train, y_train)

            # predict on test data
            y_pred_fold = best_pipe.predict(X_test)

            # try to get standard deviation of predictions
            try:
                y_pred_std_fold = best_pipe.predict(X_test, return_std=True)[1]
            except:
                y_pred_std_fold = np.zeros(y_pred_fold.shape)

            # TODO generalize this. Should be able to set other estimators then pyment and also to run without
            # compute scores per fold
            scores_fold = self._compute_scores(y_test, y_pred_fold)
            scores.append(scores_fold)

            # append results to arrays
            y_pred = np.append(y_pred, y_pred_fold)
            y_true = np.append(y_true, y_test)
            y_pred_std = np.append(y_pred_std, y_pred_std_fold)
            y_index = np.append(y_index, y_index_test)
            y_fold = np.append(y_fold, np.repeat(k, len(y_index_test)))
            k += 1

            # TODO not all models have coef_, some have feature_importances_, others can be derived using permutation importance
            # append coefficients if availabl
            coef = np.append(coef,self._get_feature_importance(best_pipe, X_train, y_train))

            # append best params
            best_params = np.append(best_params, grid.best_params_)

        # print split counts
        self._print_split_strata(data, X_train, X_test, strat_col)

        # store results
        self._results = pd.DataFrame(
            {
                "index": y_index,
                "fold": y_fold,
                "pred": y_pred,
                "std": y_pred_std,
                "true": y_true,
            }
        )

        # store scores per fold
        self._scores = pd.DataFrame(scores)

        self._overalscores = self._compute_scores(y_true, y_pred)

        # print scores
        self._print("Scores:")
        for score, val in self._overalscores.items():
            self._print(f"{score}: {val:.3f}")

        # store coefficients in dataframe if available
        # TODO: this prevents using PCA since after PCA we will have less features than before
        n_features = len(best_pipe["colselector"].feature_names_in_)
        coef = coef.reshape([-1, n_features])
        self._coef = pd.DataFrame(coef, columns=best_pipe["colselector"].feature_names_in_)

        # store best params in dataframe
        self._best_params = pd.DataFrame(best_params)

    def get_results(self):
        """Get results of training.
        Returns:
            results (pd.DataFrame): dataframe with results"""
        # create dataframe
        return self._results

    def get_scores(self):
        """Get scores.
        Returns:
            scores (dict): dictionary with scores"""
        return self._scores

    def _compute_scores(self, true, pred):
        """Get scores for model.
        Returns:
            scores (dict): dictionary with scores
        """
        # get scores
        scores = {
            "r2": r2_score(true, pred),
            "mae": mean_absolute_error(true, pred),
        }
        return scores

    def get_coefficients(self):
        """Get coefficients.
        Returns:
            coef (pd.DataFrame): dataframe with coefficients"""
        return self._coef

    def get_best_params(self):
        """Get best parameters.
        Returns:
            best_params (pd.DataFrame): dataframe with best parameters"""
        return self._best_params

    def _print_split_strata(self, total, train, test, strat_col="chron_age_group"):
        """Get split strata for train and test data.
        Args:
            total (pd.Series): total counts
            train (pd.Series): train counts
            test (pd.Series): test counts
            strat_col (str): name of column to stratify on (default: "chron_age_group")
        Returns:
            split_strata (pd.Series): split strata
        """
        count_total = total[strat_col].value_counts()
        count_train = train[strat_col].value_counts()
        count_test = test[strat_col].value_counts()

        df_splits = pd.DataFrame(
            [count_total, count_train, count_test],
            index=["total", "train", "test"],
        ).T
        self._print(df_splits)

    def _print(self, message):
        self.logger.info(message)

    def _get_feature_importance(self, pipe, X, y):
        """Get feature importance for a model

        Args:
            model (sklearn model): model to get feature importance for
            X (np.array): input data
            y (np.array): target data

        Returns:
            np.array: feature importance
        """
        model = pipe['model']
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        elif hasattr(model, "coef_"):
            return model.coef_
        else:
            return np.array([])
