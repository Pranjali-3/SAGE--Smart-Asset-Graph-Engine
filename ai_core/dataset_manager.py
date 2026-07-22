import os
import logging

import pandas as pd

from .data_processor import NASAProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetManager:

    def __init__(self):

        logger.info("Loading NASA datasets...")

        base_path = r"D:\hackathons\data\nasa\archive\CMaps"

        self.datasets = {}

        train_files = [
            "train_FD001.txt",
            "train_FD002.txt",
            "train_FD003.txt",
            "train_FD004.txt"
        ]

        for file in train_files:

            path = os.path.join(base_path, file)

            processor = NASAProcessor(path)

            processor.load_dataset()

            processor.calculate_rul()

            name = file.replace(".txt", "")

            self.datasets[name] = processor

            logger.info(f"{name} loaded.")

        logger.info("All NASA datasets loaded.")

    # =======================================================
    # Return latest engine sample
    # =======================================================

    def get_engine_sample(self, engine_id):

        for dataset_name, processor in self.datasets.items():

            df = processor.df

            if engine_id in df.engine_id.unique():

                latest_cycle = (
                    df[df.engine_id == engine_id]
                    .cycle
                    .max()
                )

                sample = processor.prepare_prediction_sample(
                    engine_id,
                    latest_cycle
                )

                actual_rul = (
                    df[
                        (df.engine_id == engine_id)
                        &
                        (df.cycle == latest_cycle)
                    ]
                    ["RUL"]
                    .iloc[0]
                )

                return {

                    "dataset": dataset_name,

                    "engine": engine_id,

                    "cycle": latest_cycle,

                    "actual_rul": actual_rul,

                    "sample": sample

                }

        return None