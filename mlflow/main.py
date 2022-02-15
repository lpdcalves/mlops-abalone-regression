"""
Creators: Luiz Paulo de C. Alves, Jonatas Rodolfo Pereira dos Santos, Ariel da Silva Alsina
Date: 13 Feb. 2022
"""
import mlflow
import os
import wandb
import hydra
from omegaconf import DictConfig

# This automatically reads in the configuration
@hydra.main(config_name='config')
def process_args(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    _ = mlflow.run(
        os.path.join(root_path, "download_data"),
        "main",
        parameters={
            "file_url": config["data"]["file_url"],
            "artifact_name": "abalone.csv",
            "artifact_type": "raw_data",
            "artifact_description": "Input data"
        }
    )

    _ = mlflow.run(
        os.path.join(root_path, "process_data"),
        "main",
        parameters={
            "input_artifact": "abalone.csv:latest",
            "artifact_name": "clean_data.csv",
            "artifact_type": "processed_data",
            "artifact_description": "Cleaned data"
        }
    )

    _ = mlflow.run(
        os.path.join(root_path, "segregate_data"),
        "main",
        parameters={
            "input_artifact": "clean_data.csv:latest",
            "artifact_root": "data",
            "artifact_type": "segregated_data",
            "test_size": config["data"]["test_size"],
            "stratify": config["data"]["stratify"],
            "random_state": config["main"]["random_seed"]
        }
    )

if __name__ == "__main__":
    process_args()
