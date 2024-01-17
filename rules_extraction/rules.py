import operator

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


class Rule(BaseEstimator):
    """
    A simple rule-based classifier.

    Parameters
    ----------
    conditions : list of str
        List of conditions defining the rule.
    label : int
        The label assigned when the conditions are met.

    Attributes
    ----------
    ops : dict
        Dictionary mapping comparison operators to corresponding functions.

    Methods
    -------
    fit(X, y=None)
        Fit the rule to the training data (not used in this basic implementation).

    predict(X)
        Predict labels for the given data.

    score(X, y)
        Calculate the accuracy score for the given data and true labels.
    """

    def __init__(self, conditions, label):
        self.conditions = conditions
        self.label = label
        self.ops = {
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
            "==": operator.eq,
            "!=": operator.ne,
        }

    def _parse_condition(self, condition):
        feature, rest = condition.split(" ", 1)
        op, threshold = rest.split(" ", 1)
        return feature, op, float(threshold)

    def fit(self, X, y=None):
        """
        Fit the rule to the training data.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            The training input samples.
        y : array-like, default=None
            Ignored.

        Returns
        -------
        self : object
            Returns an instance of the rule.
        """
        return self

    def predict(self, X):
        """
        Predict labels for the given data.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            The input samples.

        Returns
        -------
        predictions : array-like
            Array of predicted labels.
        """
        if isinstance(X, pd.Series):
            X = pd.DataFrame([X])
        elif not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError("Input must be a Pandas DataFrame or a NumPy array.")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        predictions = []
        for _, data_point in X.iterrows():
            conditions_met = all(
                self.ops[op](data_point[feature], threshold)
                for condition in self.conditions
                for feature, op, threshold in [self._parse_condition(condition)]
            )
            predictions.append(self.label if conditions_met else 1 - self.label)

        return predictions

    def score(self, X, y):
        """
        Calculate the accuracy score for the given data and true labels.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            The input samples.
        y : array-like
            The true labels.

        Returns
        -------
        accuracy : float
            The accuracy score.
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class EnsembleRule(BaseEstimator, ClassifierMixin):
    """
    A simple ensemble of rule-based classifiers.

    Parameters
    ----------
    rules : list of Rule
        List of individual rule-based classifiers.

    Methods
    -------
    fit(X, y)
        Fit the ensemble to the training data.

    predict(X)
        Predict labels for the given data.

    score(X, y)
        Calculate the accuracy score for the given data and true labels.
    """

    def __init__(self, rules):
        self.rules = rules

    def fit(self, X, y):
        """
        Fit the ensemble to the training data.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            The training input samples.
        y : array-like
            The target values.

        Returns
        -------
        self : object
            Returns an instance of the ensemble.
        """
        for rule in self.rules:
            rule.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict labels for the given data.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            The input samples.

        Returns
        -------
        predictions : array-like
            Array of predicted labels.
        """
        if isinstance(X, pd.Series):
            X = pd.DataFrame([X])
        elif not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError("Input must be a Pandas DataFrame or a NumPy array.")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        scores = np.array([rule.predict(X) for rule in self.rules]).T
        average_scores = np.mean(scores, axis=1)
        # Assign True (data point == label) when the average rule voting score is equal
        predictions = (average_scores >= 0.5).astype(int)

        return predictions

    def score(self, X, y):
        """
        Calculate the accuracy score for the given data and true labels.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            The input samples.
        y : array-like
            The true labels.

        Returns
        -------
        accuracy : float
            The accuracy score.
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class RuleRanker:
    """
    Handler for managing, applying, and evaluating rules extracted from a Random Forest model.

    :param rules: The list of rules. Each rule should be a list or a string.
    :type rules: list
    """

    ops = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "==": operator.eq,
        "!=": operator.ne,
    }

    def __init__(self, rules, X, y):
        assert all(
            isinstance(rule, tuple)
            and isinstance(rule[0], list)
            and all(isinstance(condition, str) for condition in rule[0])
            and isinstance(rule[1], int)
            and (rule[1] == 0 or rule[1] == 1)  # To ensure the integer is 0 or 1
            for rule in rules
        ), "Rules should be tuples with (list of strings, int where int is 0 or 1)"

        self.rules = rules
        self.perceptron = None
        # self.data = data
        self.X = X
        self.y = y

    @staticmethod
    def is_rule(data_point, rule):
        """
        Check whether a data point satisfies a particular rule.

        :param data_point: The data point to be checked.
        :type data_point: numpy.ndarray
        :param rule: The rule against which to check the data point.
                     Expected to be a tuple of (list, int).
        :type rule: tuple
        :return: True if the data point satisfies the rule, False otherwise.
        :rtype: bool
        """
        assert (
            isinstance(rule, tuple)
            and len(rule) == 2
            and isinstance(rule[0], list)
            and isinstance(rule[1], int)
        ), "rule should be a tuple of (list, int)"

        for rule_term in rule[0]:
            terms = rule_term.split()
            column_index = int(terms[0])
            threshold = float(terms[2])
            operation = RuleRanker.ops.get(terms[1], None)

            if operation is None:
                raise ValueError(f"Unknown operation: {terms[1]}")

            if not operation(data_point[column_index], threshold):
                return False  # Return early if any rule_term is not satisfied

        return True  # All rule_terms are satisfied

    def data_to_rules(self, X_arr):
        """
        Transform a dataset based on the set of rules, creating binary features.

        :param X_arr: The input data array.
        :type X_arr: numpy.ndarray
        :return: The transformed data array.
        :rtype: numpy.ndarray
        """

        def apply_rules(data_point):
            return [1 if self.is_rule(data_point, rule) else 0 for rule in self.rules]

        return np.apply_along_axis(apply_rules, 1, np.asarray(X_arr))

    def fit_perceptron(self, X_train, y_train, penalty="l1", alpha=0.01, **kwargs):
        """
        Fit a Perceptron model to the training data.

        :param X_train: The input training data.
        :type X_train: numpy.ndarray
        :param y_train: The target values for training data.
        :type y_train: numpy.ndarray
        :param penalty: The penalty to be used by the Perceptron model (default is 'l1').
        :type penalty: str
        :param alpha: Constant that multiplies the regularization term (default is 0.01).
        :type alpha: float
        """
        self.perceptron = Perceptron(penalty=penalty, alpha=alpha, **kwargs)
        X_train_rules = self.data_to_rules(X_train)
        self.perceptron.fit(X_train_rules, y_train)

    def rank_rules(self, N=None, penalty="l1", alpha=0.01, **kwargs):
        """
        Rank the rules based on the absolute values of Perceptron coefficients.

        :param N: Optional parameter to return the top n rules.
        :type N: int or None
        :return: A list of tuples containing rule and its absolute importance.
        :rtype: list
        :raises ValueError: If the perceptron has not been trained.
        """
        self.fit_perceptron(self.X, self.y, penalty=penalty, alpha=alpha, **kwargs)
        if self.perceptron is None or self.perceptron.coef_ is None:
            raise ValueError("The perceptron must be trained before ranking rules.")

        rule_importance = self.perceptron.coef_[0]
        absolute_importance = np.abs(rule_importance)
        sorted_indices = np.argsort(absolute_importance)[::-1]
        most_predictive_rules = [self.rules[i] for i in sorted_indices]

        return most_predictive_rules[:N] if N is not None else most_predictive_rules
