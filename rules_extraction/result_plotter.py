import matplotlib.pyplot as plt

from .rule_handler import RuleHandler


class ResultPlotter:
    def __init__(self, rules):
        self.rules = rules

    def _accuracy_N_rules(self, X_test, y_test, N):
        """
        Compute accuracy using N rules.

        :param X_test: Test data features.
        :param y_test: Test data labels.
        :param N: Maximum number of rules to consider.
        :return: Tuple (n_rules_used, scores)
        """
        scores = []
        n_rules_used = list(range(1, N + 1, 2))
        for n in n_rules_used:
            rule_handler = RuleHandler(rules=self.rules[:n])  # Limiting to n rules
            score = rule_handler.score(X_test, y_test, rule_handler.rules)
            scores.append(score)

        return n_rules_used, scores

    def plot_accuracy(self, X_test, y_test, class_name=None, N=5, save_path=None):
        """
        Plots and optionally saves a plot of accuracy vs. number of rules.

        :param X_test: test data features
        :param y_test: test data labels
        :param class_name: string, name of the class
        :param N: int, maximum number of rules to consider
        :param save_path: str, if provided, the path where the plot will be saved
        """
        n_rules_used, scores = self._accuracy_N_rules(X_test, y_test, N)

        # Plotting logic starts here
        plt.figure(figsize=(10, 6))
        plt.plot(
            n_rules_used, scores, marker="o", linestyle="-", color="b", linewidth=1
        )

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
