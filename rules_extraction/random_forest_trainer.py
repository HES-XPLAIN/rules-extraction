import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree


class RandomForestTrainer:
    """Handles training and rule extraction from a random forest classifier."""

    def __init__(self, dataset):
        """
        Initialize with a dataset. It assumes that the last column is the target.

        :param dataset: The dataset to use for training and rule extraction.
        :type dataset: pd.DataFrame
        :raises AssertionError: If dataset is not a pd.DataFrame.
        """
        assert isinstance(dataset, pd.DataFrame), "Dataset should be a pandas DataFrame"
        self.dataset = dataset
        self.model = None
        self.feature_columns = dataset.columns[
            :-1
        ]  # assuming the last column is the target
        self.target_column = dataset.columns[-1]

    def fit(self, X=None, y=None, **kwargs):
        """
        Train a random forest classifier on the dataset.

        :param X: Features for training. If None, uses self.dataset for training.
        :type X: pd.DataFrame or None
        :param y: Labels for training. If None, uses self.dataset for training.
        :type y: pd.DataFrame or None
        :param kwargs: Arguments to pass to train_test_split and RandomForestClassifier.
        :type kwargs: dict
        """
        if X is None or y is None:
            X = self.dataset[self.feature_columns]
            y = self.dataset[self.target_column]

        test_size = kwargs.pop("test_size", 0.2)
        random_state = kwargs.get("random_state", None)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model = RandomForestClassifier(**kwargs)
        self.model.fit(X_train, y_train)

    def evaluate(self, X=None, y=None):
        """
        Evaluate the model's accuracy on a test set.

        :param X: Features for evaluation.
        :type X: array-like (np.array, pd.DataFrame, list, etc.)
        :param y: Labels for evaluation.
        :type y: array-like (np.array, pd.DataFrame, list, etc.)
        :return: Accuracy of the model on the test set.
        :rtype: float
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit() before evaluate().")

        if X is None or y is None:
            raise ValueError(
                "You must provide X and y for evaluation. "
                "They should be different from the training data used to fit the model."
            )

        return self.model.score(X, y)

    @staticmethod
    def recurse(tree_, feature_name, node, current_rule, rules_list):
        """Recursively traverse the tree to extract rules."""
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # left child
            left_rule = current_rule.copy()
            left_rule.append(f"{name} <= {threshold:.2f}")
            RandomForestTrainer.recurse(
                tree_, feature_name, tree_.children_left[node], left_rule, rules_list
            )

            # right child
            right_rule = current_rule.copy()
            right_rule.append(f"{name} > {threshold:.2f}")
            RandomForestTrainer.recurse(
                tree_, feature_name, tree_.children_right[node], right_rule, rules_list
            )
        else:
            # Extract the label based on class distributions at the leaf node
            label = 0 if tree_.value[node][0][0] > tree_.value[node][0][1] else 1
            rules_list.append((current_rule, label))

    def extract_rules(self, tree):
        """Extract rules from a single decision tree."""
        feature_names = self.feature_columns
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        rules_list = []

        RandomForestTrainer.recurse(
            tree_, feature_name, 0, [], rules_list
        )  # start from the root node

        return rules_list

    def extract_all_rules(self, verbose=1):
        """
        Extract rules from all the trees in the random forest.

        :param verbose: Control the verbosity of messages printed to console.
            - 0: No output
            - 1: Print the total number of extracted rules (default)
            - 2+: Any other detailed messages, if applicable
        :type verbose: int
        :return: List of all extracted rules.
        :rtype: list
        :raises AssertionError: If model has not been trained yet.
        """
        assert self.model is not None, "Model is not trained yet"
        trees = self.model.estimators_
        rules_per_forest = []

        for tree in trees:
            rules_per_tree = self.extract_rules(tree)
            rules_per_forest.append(rules_per_tree)

        all_rules = [rule for tree_rules in rules_per_forest for rule in tree_rules]
        if verbose == 1:
            print(f"Number of rules is {len(all_rules)}")

        return all_rules
