import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import yaml


def write_error(output_file, version, message):
    """
    Write error metrics JSON.
    """
    error_data = {
        "version": version,
        "status": "error",
        "error_message": message
    }

    with open(output_file, "w") as f:
        json.dump(error_data, f, indent=2)

    return error_data


def load_and_validate_config(config_path):
    """
    Load YAML config and validate required fields.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError:
        raise ValueError("Invalid YAML format")

    if not isinstance(config, dict):
        raise ValueError("Invalid config structure")

    required_fields = ["seed", "window", "version"]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing config field: {field}")

    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")

    if not isinstance(config["window"], int) or config["window"] <= 0:
        raise ValueError("window must be a positive integer")

    if not isinstance(config["version"], str):
        raise ValueError("version must be a string")

    return config


def load_and_validate_dataset(input_path):
    """
    Load CSV and validate dataset.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError("Empty file")
    except pd.errors.ParserError:
        raise ValueError("Invalid CSV format")

    if df.empty:
        raise ValueError("Empty file")

    if "close" not in df.columns:
        raise ValueError("Missing required column: close")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="MLOps Technical Assessment"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-file", required=True)

    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    start_time = time.time()
    version = "v1"

    try:
        logging.info("========== JOB START ==========")

        # -----------------------------
        # Load Config
        # -----------------------------
        config = load_and_validate_config(args.config)

        seed = config["seed"]
        window = config["window"]
        version = config["version"]

        np.random.seed(seed)

        logging.info(
            f"Config validated: seed={seed}, "
            f"window={window}, version={version}"
        )

        # -----------------------------
        # Load Dataset
        # -----------------------------
        df = load_and_validate_dataset(args.input)

        logging.info(f"Rows loaded: {len(df)}")

        # -----------------------------
        # Rolling Mean
        # -----------------------------
        logging.info("Computing rolling mean")

        rolling_mean = df["close"].rolling(window=window).mean()

        # First (window - 1) rows produce NaN values.
        # Comparisons with NaN evaluate to False,
        # therefore signal becomes 0 for those rows.
        logging.info(
            f"First {window - 1} rows contain NaN rolling means"
        )

        # -----------------------------
        # Signal Generation
        # -----------------------------
        logging.info("Generating signals")

        signal = (df["close"] > rolling_mean).astype(int)

        # -----------------------------
        # Metrics
        # -----------------------------
        signal_rate = float(signal.mean())

        latency_ms = int(
            (time.time() - start_time) * 1000
        )

        metrics = {
            "version": version,
            "rows_processed": int(len(df)),
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": latency_ms,
            "seed": seed,
            "status": "success"
        }

        logging.info(
            f"Metrics Summary | "
            f"rows_processed={metrics['rows_processed']} | "
            f"signal_rate={metrics['value']} | "
            f"latency_ms={metrics['latency_ms']}"
        )

        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)

        logging.info("Metrics written successfully")
        logging.info("Job completed successfully")
        logging.info("========== JOB END ==========")

        print(json.dumps(metrics))

    except Exception as e:
        logging.exception("Job failed")

        error_metrics = write_error(
            args.output,
            version,
            str(e)
        )

        print(json.dumps(error_metrics))

        # Non-zero exit code for Docker
        raise

    finally:
        logging.info("Program terminated")


if __name__ == "__main__":
    main()
