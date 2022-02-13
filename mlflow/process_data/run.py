"""
Creators: Luiz Paulo de C. Alves, Jonatas Rodolfo Pereira dos Santos, Ariel da Silva Alsina
Date: 13 Feb. 2022
"""
import argparse
import logging
import seaborn as sns
import pandas as pd
import wandb
from pandas_profiling import ProfileReport

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

def process_args(args):

    run = wandb.init(job_type="process_data")

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    abalone = pd.read_csv(
        artifact_path
    )

    # iris = pd.read_csv(
    #     artifact_path,
    #     skiprows=1,
    #     names=("sepal_length", "sepal_width", "petal_length", "petal_width", "target"),
    # )

    profile = ProfileReport(abalone, title="Pandas Profiling Report", explorative=True)
    profile.to_file("abalone_profile.html")

    # logger.info("Creating profile artifact")

    # profile_artifact = wandb.Artifact(
    #     name="abalone_profile.html",
    #     type=process_data,
    #     description="Abalone dataset auto profiling done by PandasProfiling",
    # )
    # profile_artifact.add_file("abalone_profile.html")
    # logger.info("Logging profile artifact")
    # run.log_artifact(profile_artifact)

    logger.info("Creating artifact")

    abalone.to_csv("clean_data.csv")

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file("clean_data.csv")

    logger.info("Logging artifact")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to W&B",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    ARGS = parser.parse_args()

    process_args(ARGS)

