import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
    def generate_failure_labels(self):

        """
        Generate failure labels from Remaining Useful Life.
        """

        logger.info("Generating failure labels...")

        if self.df is None:
            raise ValueError("Dataset not loaded.")

        def label(rul):

            if rul > 100:
                return "Healthy"

            elif rul > 40:
                return "Warning"

            else:
                return "Critical"

        self.df["failure_label"] = self.df["RUL"].apply(label)

        logger.info("Failure labels generated.")

        return self.df
    def calculate_rolling_features(self):

        """
        Calculate rolling mean and rolling standard deviation
        for every sensor.
        """

        logger.info("Calculating rolling sensor features...")

        if self.df is None:
            raise ValueError("Dataset not loaded.")

        sensor_columns = [

            f"sensor_{i}"

            for i in range(1, 22)

        ]

        for sensor in sensor_columns:

            self.df[f"{sensor}_mean"] = (

                self.df
                .groupby("engine_id")[sensor]
                .transform(

                    lambda x:

                    x.rolling(
                        window=5,
                        min_periods=1
                    ).mean()

                )

            )

            self.df[f"{sensor}_std"] = (

                self.df
                .groupby("engine_id")[sensor]
                .transform(

                    lambda x:

                    x.rolling(
                        window=5,
                        min_periods=1
                    ).std()

                )

                .fillna(0)

            )

        logger.info("Rolling features calculated.")

        return self.df
    def detect_sensor_trends(self):
        """
        Detect sensor trends using rolling means.
        """

        logger.info("Detecting sensor trends...")

        if self.df is None:
            raise ValueError("Dataset not loaded.")

        sensor_columns = [
            f"sensor_{i}"
            for i in range(1, 22)
        ]

        for sensor in sensor_columns:

            mean_col = f"{sensor}_mean"

            trend_col = f"{sensor}_trend"

            trend_diff = (

                self.df
                .groupby("engine_id")[mean_col]
                .diff()

            )

            self.df[trend_col] = np.where(

                trend_diff > 0.01,
                "Increasing",

                np.where(
                    trend_diff < -0.01,
                    "Decreasing",
                    "Stable"
                )
            )

        logger.info("Sensor trends detected.")

        return self.df
    def detect_anomalies(self):
        """
        Detect anomalies using rolling statistics.
        """

        logger.info("Detecting anomalies...")

        if self.df is None:
            raise ValueError("Dataset not loaded.")

        sensor_columns = [
            f"sensor_{i}"
            for i in range(1, 22)
        ]

        for sensor in sensor_columns:

            mean_col = f"{sensor}_mean"
            std_col = f"{sensor}_std"
            anomaly_col = f"{sensor}_status"

            deviation = abs(
                self.df[sensor] - self.df[mean_col]
            )

            self.df[anomaly_col] = np.where(
                deviation > (2 * self.df[std_col]),
                "Critical",
                np.where(
                    deviation > self.df[std_col],
                    "Warning",
                    "Normal"
                )
            )

        logger.info("Anomaly detection completed.")

        return self.df
    def prepare_ml_dataset(self):
        """
        Prepare dataset for Machine Learning.
        """

        logger.info("Preparing ML dataset...")

        if self.df is None:
            raise ValueError("Dataset not loaded.")

        drop_columns = [
            "engine_id",
            "cycle",
            "failure_label"
        ]

        trend_columns = [
            col for col in self.df.columns
            if col.endswith("_trend")
        ]

        status_columns = [
            col for col in self.df.columns
            if col.endswith("_status")
        ]

        feature_df = self.df.drop(
            columns=drop_columns + trend_columns + status_columns
        )

        X = feature_df.drop(columns=["RUL"])

        y = feature_df["RUL"]

        logger.info("ML dataset prepared.")

        return X, y
    def split_dataset(self, X, y):

        logger.info("Splitting dataset...")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        logger.info("Dataset split completed.")

        return X_train, X_test, y_train, y_test


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

    df = processor.generate_failure_labels()
    df = processor.calculate_rolling_features()
    df = processor.detect_sensor_trends()
    df = processor.detect_anomalies()

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

    print()
    print("Remaining Useful Life (RUL)")
    print(
        df[
            [
                "engine_id",
                "cycle",
                "health_index",
                "RUL"
            ]
        ].head(20)
    )

    print()
    print("Failure Label Distribution")
    print(
        df["failure_label"].value_counts()
    )

    print()
    print("Rolling Features")
    print(
        df[
            [
                "engine_id",
                "cycle",
                "sensor_2",
                "sensor_2_mean",
                "sensor_2_std"
            ]
        ].head(20)
    )

    print()
    print("Sensor Trend Detection")
    print(
        df[
            [
                "engine_id",
                "cycle",
                "sensor_2",
                "sensor_2_mean",
                "sensor_2_trend"
            ]
        ].head(25)
    )
    print()
print("Sensor Anomaly Detection")
print(
    df[
        [
            "engine_id",
            "cycle",
            "sensor_2",
            "sensor_2_mean",
            "sensor_2_std",
            "sensor_2_status"
        ]
    ].head(25)
)
X, y = processor.prepare_ml_dataset()

print()
print("Feature Matrix Shape")
print(X.shape)

print()
print("Target Shape")
print(y.shape)

print()
print(X.head())
X, y = processor.prepare_ml_dataset()

X_train, X_test, y_train, y_test = processor.split_dataset(X, y)

print()
print("Training Features:", X_train.shape)
print("Testing Features :", X_test.shape)

print()

print("Training Labels :", y_train.shape)
print("Testing Labels  :", y_test.shape)