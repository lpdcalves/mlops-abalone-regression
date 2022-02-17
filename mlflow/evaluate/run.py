"""
Creators: Luiz Paulo de C. Alves, Jonatas Rodolfo Pereira dos Santos, Ariel da Silva Alsina
Date: 13 Feb. 2022
"""
import argparse
import logging
import pandas as pd
import wandb
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

def process_args(args):
    
    run = wandb.init(job_type="test")

    logger.info("Downloading and reading test artifact")
    test_data_path = run.use_artifact(args.test_data).file()
    df_test = pd.read_csv(test_data_path)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    x_test = df_test.copy()
    y_test = x_test.pop("Age")
    
    # Encoding the target variable
    logger.info("Encoding Target Variable")
    
    ## Download inference artifact
    logger.info(f"Downloading and reading the exported model {args.model_export}")
    model_export_path = run.use_artifact(args.model_export).download()

    ## Load the inference pipeline
    pipe = mlflow.sklearn.load_model(model_export_path)

    ## Predict test data
    predict = pipe.predict(x_test)

    # Evaluation Metrics
    logger.info("Evaluation metrics")
    # Metric: Max Error
    erro_maximo = max_error(y_test, predict)
    run.summary["max_error"] = erro_maximo
    
    logger.info(f"Max Error: {erro_maximo}")

    # Metric: Mean Square Error
    erro_medio_quadratico = mean_squared_error(y_test, predict)
    run.summary["mean_squared_error"] = erro_medio_quadratico

    logger.info(f"Mean Square Error: {erro_medio_quadratico}")

    # Metric: Mean Absolute Erro
    erro_medio_absoluto = mean_absolute_error(y_test, predict)
    run.summary["mean_absolute_error"] = erro_medio_absoluto

    logger.info(f"Mean Absolute Error: {erro_medio_absoluto}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model on the test artifact",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate",
        required=True,
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data",
        required=True,
    )

    ARGS = parser.parse_args()

    process_args(ARGS)