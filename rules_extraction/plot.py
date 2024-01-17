import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .rules import EnsembleRule


def plot_accuracy(rules, df_test, class_name=None, n=5, save_path=None):
    """
    Plots and optionally saves a plot of accuracy vs. number of rules.

    :param X_test: test data features
    :param y_test: test data labels
    :param class_name: string, name of the class
    :param N: int, maximum number of rules to consider
    :param save_path: str, if provided, the path where the plot will be saved
    """

    n_rules_used = list(range(1, n))
    scores = []
    X = df_test.drop(columns=["path", "label", "binary_label"])
    y = df_test["binary_label"]

    for m in n_rules_used:
        ensemble_rule = EnsembleRule(rules[:m])
        score = ensemble_rule.score(X, y)
        scores.append(score)

    # Plotting logic starts here
    plt.figure(figsize=(10, 6))
    plt.plot(n_rules_used, scores, marker="o", linestyle="-", color="b", linewidth=1)

    # Adding titles and labels
    plt.title("Accuracy vs. Number of Rules Used")
    plt.xlabel("Number of Top Rules Selected")
    plt.ylabel("Accuracy")

    # Adjusting the x-axis labels
    plt.xticks(n_rules_used, [f"Top {n}" for n in n_rules_used])

    # Optionally adding class name to the plot
    if class_name:
        plt.legend([f"Class: {class_name}"])

    # Adding a grid with darker gray lines
    plt.grid(True, linestyle="--", alpha=0.7, color="#a0a0a0")

    # Setting background color to very light grey
    plt.gca().set_facecolor("#f0f0f0")

    # Save the plot if a save path is provided
    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Plot saved at: {save_path}")
        except Exception as e:
            raise RuntimeError(f"Couldn't save the plot due to: {str(e)}") from e

    # Displaying the plot
    plt.show()


def transform():
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    return transform


def plot_frontier(
    df, rule, target_class, model=None, alpha=0.65, save_path=None, device=None
):
    """
    Plots and optionally saves a plot showing one rule frontier and embedded images.

    :param df: data that stores image label and path
    :param rule: rule you want to plot, should be a Rule object
    :param target_class: string, name of the class
    :param model: torch model you used
    :param alpha: float between 0 and 1, transparency level
    :param save_path: str, if provided, the path where the plot will be saved
    """

    df.columns = df.columns.astype(str)
    condition_0, condition_1 = rule.conditions
    feature_0, op_0, threshold_0 = rule._parse_condition(condition_0)
    feature_1, op_1, threshold_1 = rule._parse_condition(condition_1)

    row_target = df[df["label"] == target_class].index.tolist()
    row_non_target = (
        df[df["label"] != target_class]
        .sample(n=len(row_target), random_state=1)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for row in row_target + row_non_target:
        img_path = df.loc[row, "path"]
        img = Image.open(img_path)
        if model:
            img = img.resize((100, 100))  # Resizing the image to 100x100 pixels

            img_tensor = transform()(img).unsqueeze(0).to(device)
            feature_maps = model.features(img_tensor)

            # Extract specific feature maps
            feature_map_0 = (
                feature_maps[0, int(feature_0), :, :].cpu().detach().numpy()
            )  # Assuming PyTorch tensor
            feature_map_1 = (
                feature_maps[0, int(feature_1), :, :].cpu().detach().numpy()
            )  # Assuming PyTorch tensor

            combined_feature_map = (
                feature_map_0 + feature_map_1
            )  # Combine the feature maps

        resized_feature_map = np.array(
            Image.fromarray(combined_feature_map).resize((224, 224), Image.BILINEAR)
        )
        normalized_map = (resized_feature_map - resized_feature_map.min()) / (
            resized_feature_map.max() - resized_feature_map.min()
        )
        heatmap = Image.fromarray(np.uint8(255 * normalized_map))
        heatmap = heatmap.resize(img.size, Image.BILINEAR)
        overlay = Image.blend(
            img, heatmap.convert("RGB"), alpha=alpha
        )  # Adjust alpha for overlay intensity

        scale = 0.003  # Adjust the scale factor as needed
        ax.imshow(
            overlay,
            extent=(
                df.loc[row, feature_0] - scale * 50,
                df.loc[row, feature_0] + scale * 50,
                df.loc[row, feature_1] - scale * 50,
                df.loc[row, feature_1] + scale * 50,
            ),
            aspect="auto",
        )

    # Rest of your code for setting x and y limits, plotting lines, and other plot configurations
    ax.set_xlim([0, df.loc[:, feature_0].max()])
    ax.set_ylim([0, df.loc[:, feature_1].max()])

    # Calculate the fractions of the plot area for threshold_0 and threshold_1
    x_frac = (threshold_0 - 0) / df.loc[:, feature_0].max()
    y_frac = (threshold_1 - 0) / df.loc[:, feature_1].max()

    # Plotting lines from the threshold values
    if (op_0 == ">" or op_0 == ">=") and (op_1 == ">" or op_1 == ">="):
        ax.axvline(
            x=threshold_0,
            ymin=y_frac,
            ymax=1,
            color="cyan",
            linestyle="-",
            linewidth=2,
        )
        ax.axhline(
            y=threshold_1,
            xmin=x_frac,
            xmax=1,
            color="cyan",
            linestyle="-",
            linewidth=2,
        )
    elif (op_0 == ">" or op_0 == ">=") and (op_1 == "<" or op_1 == "<="):
        ax.axvline(
            x=threshold_0,
            ymin=0,
            ymax=y_frac,
            color="cyan",
            linestyle="-",
            linewidth=2,
        )
        ax.axhline(
            y=threshold_1,
            xmin=x_frac,
            xmax=1,
            color="cyan",
            linestyle="-",
            linewidth=2,
        )
    elif (op_0 == "<" or op_0 == "<=") and (op_1 == ">" or op_1 == ">="):
        ax.axvline(
            x=threshold_0,
            ymin=y_frac,
            ymax=1,
            color="cyan",
            linestyle="-",
            linewidth=2,
        )
        ax.axhline(
            y=threshold_1,
            xmin=0,
            xmax=x_frac,
            color="cyan",
            linestyle="-",
            linewidth=2,
        )
    elif (op_0 == "<" or op_0 == "<=") and (op_1 == "<" or op_1 == "<="):
        ax.axvline(
            x=threshold_0,
            ymin=0,
            ymax=y_frac,
            color="cyan",
            linestyle="-",
            linewidth=2,
        )
        ax.axhline(
            y=threshold_1,
            xmin=0,
            xmax=x_frac,
            color="cyan",
            linestyle="-",
            linewidth=2,
        )

    ax.set_xlabel(f"Feature {feature_0} average feature activation -->", fontsize=12)
    ax.set_ylabel(f"Feature {feature_1} average activation -->", fontsize=12)

    # Hide the plot's edges
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Save the plot if a save path is provided
    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Plot saved at: {save_path}")
        except Exception as e:
            raise RuntimeError(f"Couldn't save the plot due to: {str(e)}") from e

    plt.show()
