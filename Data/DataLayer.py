import pandas as pd
from Data.CreateSyntheticData import (
    createMoons, createHelix, createSinusoidalData,
    createSpiral, createBlobs, createCircles
)
from tkinter import messagebox


class DataLoader:

    @staticmethod
    def load_data(file_path):
        # Identify file format and load accordingly
        if file_path.endswith(".csv"):
            dataset = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            dataset = pd.read_excel(file_path)
        elif file_path.endswith(".json"):
            dataset = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")

        # Ensure the dataset contains a column with ground truth labels
        if "outlier" not in dataset.columns:
            raise ValueError("Dataset must contain 'outlier' column for true labels")

        # Separate features (X) and labels (y)
        y = dataset["outlier"].values
        X = dataset.drop(['outlier'], axis=1).values

        # Return preview of the dataset
        return X, y, dataset

    @staticmethod
    def create_SyntheticData(choice):
        print("---------")
        X, y = None, None

        if choice == "moons":
            X, y = createMoons()
        elif choice == "circle":
            X, y = createCircles()
        elif choice == "blobs":
            X, y = createBlobs()
        elif choice == "spiral":
            X, y = createSpiral()
        elif choice == "sin":
            X, y = createSinusoidalData()
        elif choice == "helix":
            X, y = createHelix()
        else:
            messagebox.showerror("Error", "Unknown dataset type selected.")

        return X, y
