import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree


def compute_features(model, loader, class_dict, device):
    # here loader can be train, test, filtered or not

    features_list, labels_list, paths_list = [], [], []

    for images, labels, path in loader:
        paths = list(path)
        images, labels = images.to(device), labels.to(device)
        features = extract_features_vgg(model, images)
        features_list.extend(features.tolist())
        labels_list.extend(labels.tolist())
        paths_list.extend(paths)

    df = pd.DataFrame(features_list)
    if class_dict is not None:
        labels_list = [class_dict[str(item)] for item in labels_list]
    df["label"] = labels_list
    df["path"] = paths_list

    # create a df with all features stored from train or test dataset
    # df.to_csv("./features_map.csv", index=False)
    return df


def filter_dataset(model, loader, device):
    """
    Use a loader and a model and return the list of correct predicted datapoint index in the loader by the model.
    This function allow to create a filtered loader using the index list
    """
    correct_indices_global = []

    model = model.eval()
    for i, (image, label, image_path) in enumerate(loader):
        image, label = image.to(device), label.to(device)
        with torch.no_grad():
            logits = model(image)
            predictions = torch.argmax(logits, dim=1)
            correct_local = (
                (predictions == label).nonzero(as_tuple=False).squeeze().cpu().numpy()
            )

            # If correct_local is a scalar, convert it to an array for consistency.
            if correct_local.ndim == 0:
                correct_local = np.array([correct_local])

            # Convert local batch indices to global indices.
            correct_global = i * loader.batch_size + correct_local
            correct_indices_global.extend(correct_global)

    return correct_indices_global


def extract_features_vgg(model, x):
    """
    Predefined feature extraction for VGG-like models.

    Parameters
    ----------
    x : torch.Tensor
        input data tensor

    Returns
    -------
    torch.Tensor
        extracted features
    """
    return torch.mean(model.features(x), dim=[2, 3])


def extract_features_resnet(x):
    """
    Predefined feature extraction for ResNet-like models. [NOT IMPLEMENTED]

    Parameters
    ----------
    x : torch.Tensor
        input data tensor
    """
    pass


def make_target_df(df_features, target_class):
    """
    Produces a DataFrame with binary labels: 1 for `target_class` and 0 for other classes.

    Parameters
    ----------
    df : pd.DataFrame
        input DataFrame
    target_class : int or str
        class label to be considered as target (1)

    Returns
    -------
    pd.DataFrame
        new DataFrame with binary labels
    """

    # Extract all rows where label matches the target_class
    target_df = df_features[df_features["label"] == target_class]
    n = target_df.shape[0]

    # Extract randomly n rows where label doesn't match target_class
    non_target_df = df_features[df_features["label"] != target_class].sample(
        n, random_state=1
    )

    final_df = pd.concat([target_df, non_target_df])
    final_df["binary_label"] = np.where(final_df["label"] == target_class, 1, 0)
    final_df.columns = final_df.columns.astype(str)

    return final_df


def recurse(tree_, feature_name, node, current_rule, rules_list):
    """Recursively traverse the tree to extract rules."""
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree_.threshold[node]

        # left child
        left_rule = current_rule.copy()
        left_rule.append(f"{name} <= {threshold:.2f}")
        recurse(tree_, feature_name, tree_.children_left[node], left_rule, rules_list)

        # right child
        right_rule = current_rule.copy()
        right_rule.append(f"{name} > {threshold:.2f}")
        recurse(tree_, feature_name, tree_.children_right[node], right_rule, rules_list)
    else:
        # Extract the label based on class distributions at the leaf node
        label = 0 if tree_.value[node][0][0] > tree_.value[node][0][1] else 1
        rules_list.append((current_rule, label))


def extract_rules(tree, feature_columns):
    """Extract rules from a single decision tree."""
    feature_names = feature_columns
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules_list = []

    recurse(tree_, feature_name, 0, [], rules_list)  # start from the root node

    return rules_list


def extract_all_rules(model, X, y, **kwargs):
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
    rf = RandomForestClassifier(**kwargs)
    rf.fit(X, y)
    trees = rf.estimators_
    rules_per_forest = []

    for tree in trees:
        rules_per_tree = extract_rules(tree, X.columns)
        rules_per_forest.append(rules_per_tree)

    all_rules = [rule for tree_rules in rules_per_forest for rule in tree_rules]
    # print(f"Number of rules is {len(all_rules)}")

    return all_rules
