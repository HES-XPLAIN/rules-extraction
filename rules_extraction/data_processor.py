import os

import numpy as np
import pandas as pd
import torch


class DataProcessor:
    def __init__(self, model, dataloader, device=torch.device("cuda")):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def filter_dataset(self):
        pass

    @staticmethod
    def make_target_df(df, target_class):
        # Extract all rows where label matches the target_class
        target_df = df[df["label"] == target_class]
        n = target_df.shape[0]

        # Extract randomly n rows where label doesn't match target_class
        non_target_df = df[df["label"] != target_class].sample(n)

        final_df = pd.concat([target_df, non_target_df])
        final_df["binary_label"] = np.where(final_df["label"] == target_class, 1, 0)

        return final_df

    def process_dataset(self, feature_layer, target_class):
        self.model.to(self.device)
        features_list, labels_list, paths_list = [], [], []

        for images, labels, path in self.dataloader:
            paths = list(path)
            images, labels = images.to(self.device), labels.to(self.device)
            # NEED TO IMPLEMENT FEATURE_LAYER
            features = self.model.features(images)
            avg_features = torch.mean(features, dim=[2, 3])
            features_list.extend(avg_features.tolist())
            labels_list.extend(labels.tolist())
            paths_list.extend(paths)

        df = pd.DataFrame(features_list)
        df["label"] = labels_list
        df["path"] = paths_list

        # NEED TO ADD CHECK ABOUT DATA TYPE IN DF and LABEL MAPPING ETC.

        folder = "binary_dataset"
        os.makedirs(folder, exist_ok=True)  # This line ensures the folder exists
        df_new = self.make_target_df(df=df, target_class=target_class)
        path = os.path.join(
            folder, f"{target_class}.csv"
        )  # This line constructs the path using os.path.join
        df_new.to_csv(path)

        # Notify the user
        print(
            f"Your new data, with target class '{target_class}', has been created and saved to: {path}"
        )
