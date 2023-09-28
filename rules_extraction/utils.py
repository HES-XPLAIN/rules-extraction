from sklearn.tree import _tree
import operator
import numpy as np

def extract_rules(tree, feature_names):
    """
    This function extracts the decision rules from a trained decision tree in a
    human-readable format. The extracted rules are returned as a list, each element
    being a tuple containing the rule and the label that would be assigned.

    Parameters:
    ----------
    tree : sklearn.tree.DecisionTreeClassifier
        A trained decision tree classifier.
    feature_names : list
        A list of strings containing the name of each feature.

    Returns:
    -------
    rules_list : list
        A list of tuples. Each tuple contains a list of strings forming the rule,
        and an integer representing the label that would be assigned by the rule.
    """

    # Access the internal tree structure from sklearn's tree object
    tree_ = tree.tree_

    # Create a list of feature names, replacing undefined features with the string "undefined!"
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    # Initialize an empty list to store the extracted rules
    rules_list = []

    def recurse(node, current_rule):
        """
        Recursive function to traverse the tree and extract rules.

        Parameters:
        ----------
        node : int
            The node id within the tree.
        current_rule : list
            A list of strings forming the current rule being constructed.
        """
        # Check if the current node is not a leaf node
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # Get the feature name and threshold value for the current node
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # left child
            left_rule = current_rule.copy()
            left_rule.append(f"{name} <= {threshold:.2f}")
            recurse(tree_.children_left[node], left_rule)

            # right child
            right_rule = current_rule.copy()
            right_rule.append(f"{name} > {threshold:.2f}")
            recurse(tree_.children_right[node], right_rule)
        else:
            # The node is a leaf node. Determine the label based on the class distribution
            # at the leaf node and append the rule and label to the `rules_list`
            label = 0 if tree_.value[node][0][0] > tree_.value[node][0][1] else 1
            rules_list.append((current_rule, label))

    # Start the recursion from the root node (node 0) with an empty rule
    recurse(0, [])
    return rules_list


def data_to_rules(X_arr, rules):
    """
    This function applies a set of rules to a dataset and transforms the data points
    based on these rules. Each rule is evaluated on each data point, and a new binary
    feature is created for each rule indicating whether the data point satisfies the rule.

    Parameters:
    ----------
    X_arr : numpy.ndarray
        The input data array with shape (n_samples, n_features).
    rules : list
        A list of tuples, where each tuple contains a rule and a label. Each rule is a
        list of strings, where each string represents a condition on a feature.

    Returns:
    -------
    transformed_data : numpy.ndarray
        A binary array of shape (n_samples, n_rules) indicating whether each data point
        satisfies each rule.
    """

    def is_rule(data_point, rule):
        """
        Helper function to check if a data point satisfies a rule.

        Parameters:
        ----------
        data_point : numpy.ndarray
            A single data point.
        rule : list
            A list of strings representing conditions on the features.

        Returns:
        -------
        bool
            True if the data point satisfies all conditions in the rule, False otherwise.
        """
        ops = {
            '<': operator.lt,
            '<=': operator.le,
            '>': operator.gt,
            '>=': operator.ge,
            '==': operator.eq,
            '!=': operator.ne
        }

        for ru in rule:
            terms = ru.split()
            column_index = int(terms[0])
            threshold = float(terms[2])
            operation = ops[terms[1]]
            if not operation(data_point[column_index], threshold):
                return False
        return True

    transformed_data = []
    for i in range(X_arr.shape[0]):
        data_point = X_arr[i, :]
        transformed_data_point = [1 if is_rule(data_point, rule[0]) else 0 for rule in rules]
        transformed_data.append(transformed_data_point)

    return np.asarray(transformed_data)


def classify_with_rules(data_point, top_rules):
    """
    Classify a single data point based on a set of rules.

    Each rule in `top_rules` consists of a set of conditions and a label.
    The function evaluates each rule on the data point, and tallies votes
    for the labels based on which rules the data point satisfies.
    The function then returns the label with the most votes.

    Parameters:
    ----------
    data_point : numpy.ndarray
        The input data point with shape (n_features,).
    top_rules : list
        A list of tuples. Each tuple contains a list of strings forming a rule,
        and an integer representing the label assigned by the rule.

    Returns:
    -------
    int
        The predicted label for the data point, either 0 or 1.
    """
    ops = {
        '<': operator.lt,
        '<=': operator.le,
        '>': operator.gt,
        '>=': operator.ge,
        '==': operator.eq,
        '!=': operator.ne
    }

    votes = {0: 0, 1: 0}

    for rule_conditions, rule_label in top_rules:
        rule_holds = all(
            ops[condition.split()[1]](data_point[int(condition.split()[0])], float(condition.split()[2]))
            for condition in rule_conditions
        )

        if rule_holds:
            votes[rule_label] += 1
        else:
            votes[1 - rule_label] += 1  # Vote for the opposite label

    # Return the label with the most votes. Handle ties by favoring label 0.
    return 0 if votes[0] >= votes[1] else 1


