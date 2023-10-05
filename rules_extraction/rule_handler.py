import json
import operator

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


class RuleHandler:
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

    def __init__(self, rules):
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
            operation = RuleHandler.ops.get(terms[1], None)

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

    def evaluate_perceptron(self, X_test, y_test, **kwargs):
        """
        Evaluate the Perceptron model on test data.

        :param X_test: The input test data.
        :type X_test: numpy.ndarray
        :param y_test: The target values for test data.
        :type y_test: numpy.ndarray
        :return: The accuracy of the Perceptron model on the test data.
        :rtype: float
        """
        X_test_rules = self.data_to_rules(X_test)
        accuracy = self.perceptron.score(X_test_rules, y_test, **kwargs)
        return accuracy

    def rank_rules(self, N=None):
        """
        Rank the rules based on the absolute values of Perceptron coefficients.

        :param N: Optional parameter to return the top n rules.
        :type N: int or None
        :return: A list of tuples containing rule and its absolute importance.
        :rtype: list
        :raises ValueError: If the perceptron has not been trained.
        """
        if self.perceptron is None or self.perceptron.coef_ is None:
            raise ValueError("The perceptron must be trained before ranking rules.")

        rule_importance = self.perceptron.coef_[0]
        absolute_importance = np.abs(rule_importance)
        sorted_indices = np.argsort(absolute_importance)[::-1]
        most_predictive_rules = [self.rules[i] for i in sorted_indices]

        return most_predictive_rules[:N] if N is not None else most_predictive_rules

    def predict(self, data, top_rules):
        """
        Classifies data points using the specified rules.

        :param data: The data to be classified.
        :type data: ndarray or 1D array-like
        :param top_rules: The rules to be used for classification.
        :type top_rules: list of tuples
        :return: The predicted labels.
        :rtype: list
        """
        data = np.array(data)
        if len(np.shape(data)) == 1:
            # Single data point
            return self._classify_data_point(data, top_rules)
        else:
            return np.array(
                [
                    self._classify_data_point(data_point, top_rules)
                    for data_point in data
                ]
            )

    @classmethod
    def _classify_data_point(cls, data_point, top_rules):
        """
        Classifies a single data point using the specified rules.

        :param data_point: The data point to be classified.
        :type data_point: 1D array-like
        :param top_rules: The rules to be used for classification.
        :type top_rules: list of tuples
        :return: The predicted label.
        :rtype: int
        """

        # Initializing the vote counter
        votes = {0: 0, 1: 0}

        for rule_conditions, rule_label in top_rules:
            rule_holds = True
            for condition in rule_conditions:
                terms = condition.split()
                column_index = int(terms[0])
                threshold = float(terms[2])
                operation = cls.ops[terms[1]]

                # Check if the data_point meets the condition
                if not operation(data_point[column_index], threshold):
                    rule_holds = False
                    break  # Exit the condition loop as soon as one condition is not met

            # Vote counting logic
            if rule_holds:
                # Voting for rule_label if rule holds
                votes[rule_label] += 1
            else:
                # Voting for the opposite of rule_label if rule does not hold
                votes[1 - rule_label] += 1

        # Returning the label with the most votes. In case of a tie, select label 0.
        return 0 if votes[0] >= votes[1] else 1

    def score(self, X_test, y_test, top_rules=None):
        """
        Computes the accuracy of classification on a given dataset.

        :param X_test: The feature matrix.
        :type X_test: ndarray or DataFrame
        :param y_test: The true labels.
        :type y_test: 1D array-like
        :param top_rules: The rules to be used for classification.
        :type top_rules: list of tuples or None
        :return: The accuracy on the given dataset.
        :rtype: float
        """
        if top_rules is None:
            raise ValueError(
                "top_rules must be provided, use rank_rules method to compute them."
            )

        y_pred = self.predict(X_test, top_rules)
        return accuracy_score(y_test, y_pred)

    def save(self, path, rules=None):
        """
        Save rules to a file.

        :param path: The path of the file to save rules to.
        :param rules: The rules to save. If None, saves self.rules. Optional, default is None.
        """
        if rules is None:
            rules = self.rules
        self.save_rules(rules, path)

    def load(self, path):
        """
        Load rules from a file and update self.rules.

        :param path: The path of the file to load rules from.
        """
        self.rules = self.load_rules(path)

    @staticmethod
    def save_rules(rules, path):
        """
        Save rules to a file.

        :param rules: The rules to save.
        :param path: The path of the file to save rules to.
        """
        with open(path, "w") as file:
            json.dump(rules, file)

    @staticmethod
    def load_rules(path):
        """
        Load rules from a file without altering self.rules.

        :param path: The path of the file to load rules from.
        :return: The loaded rules.
        """
        with open(path, "r") as file:
            return json.load(file)

    def visualize(self, rules):
        pass
