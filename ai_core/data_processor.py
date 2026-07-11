import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class NASAProcessor:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        self.columns = [
            "engine_id",
            "cycle",
            "setting1",
            "setting2",
            "setting3"
        ]

        for i in range(1, 22):
            self.columns.append(f"sensor_{i}")

        self.df = None

        logger.info("NASA Processor initialized.")

    def load_dataset(self):

        logger.info("Loading NASA dataset...")

        self.df = pd.read_csv(
            self.dataset_path,
            sep=r"\s+",
            header=None,
            names=self.columns
        )

        logger.info(f"Loaded {len(self.df)} rows.")

        return self.df
    
    def clean_dataset(self):
        """Clean NASA dataset."""

        logger.info("Cleaning dataset...")

        if self.df is None:
            raise ValueError("Dataset not loaded.")

        # Remove duplicate rows
        before = len(self.df)

        self.df.drop_duplicates(inplace=True)

        after = len(self.df)

        logger.info(f"Removed {before - after} duplicate rows.")

        # Remove rows with missing values
        before = len(self.df)

        self.df.dropna(inplace=True)

        after = len(self.df)

        logger.info(f"Removed {before - after} rows with missing values.")

        # Reset index
        self.df.reset_index(drop=True, inplace=True)

        logger.info("Dataset cleaned.")

        return self.df
    
    def normalize_dataset(self):

        f"""
        Normalize all sensor values between 0 and 1.
        """

        logger.info("Normalizing sensor values...")

        if self.df is None:
            raise ValueError("Dataset not loaded.")

        sensor_columns = [

            f"sensor_{i}"

            for i in range(1, 22)

        ]

        scaler = MinMaxScaler()

        self.df[sensor_columns] = scaler.fit_transform(

            self.df[sensor_columns]

        )

        logger.info("Sensor normalization completed.")

        return self.df
    def calculate_health_index(self):

        """
        Calculate Health Index for every engine cycle.
        """

        logger.info("Calculating Health Index...")

        if self.df is None:
            raise ValueError("Dataset not loaded.")

        sensor_columns = [

            f"sensor_{i}"

            for i in range(1, 22)

        ]

        self.df["health_index"] = (

            1 -

            self.df[sensor_columns].mean(axis=1)

        )

        logger.info("Health Index calculated.")

        return self.df

        def calculate_rul(self):
            """
            Calculate Remaining Useful Life (RUL)
            for every engine cycle.
            """

            logger.info("Calculating Remaining Useful Life...")

            if self.df is None:
                raise ValueError("Dataset not loaded.")

            max_cycles = (

                self.df
                .groupby("engine_id")["cycle"]
                .max()

            )

            self.df["max_cycle"] = (

                self.df["engine_id"]
                .map(max_cycles)

            )

            self.df["RUL"] = (

                self.df["max_cycle"]
                -

                self.df["cycle"]

            )

            self.df.drop(

                columns=["max_cycle"],

                inplace=True

            )

            logger.info("RUL calculation completed.")

            return self.df


if __name__ == "__main__":

    print("Program started")

    processor = NASAProcessor(
        r"data/nasa/archive/CMaps/train_FD001.txt"
    )

    # Load
    df = processor.load_dataset()

    # Clean
    df = processor.clean_dataset()

    # Normalize
    df = processor.normalize_dataset()

    # Health Index
    df = processor.calculate_health_index()
    df = processor.calculate_rul()

    print()
    print("First 5 Rows")
    print(df.head())

    print()
    print("Dataset Shape")
    print(df.shape)

    print()
    print("Dataset Info")
    print(df.info())

    print()
    print("Missing Values")
    print(df.isnull().sum())

    print()
    print("Normalized Sensor Values")
    print(
        df[
            [
                "sensor_1",
                "sensor_2",
                "sensor_3",
                "sensor_4",
                "sensor_5"
            ]
        ].head()
    )

    print()
    print("Health Index")
    print(
        df[
            [
                "engine_id",
                "cycle",
                "health_index"
            ]
        ].head(15)
    )

    print()
    print("Health Index Statistics")
    print(df["health_index"].describe())

    print()

print("Remaining Useful Life (RUL)")

print(

    df[[
        "engine_id",
        "cycle",
        "health_index",
        "RUL"
    ]].head(20)

)