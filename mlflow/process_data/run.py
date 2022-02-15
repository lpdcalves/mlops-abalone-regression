"""
Creators: Luiz Paulo de C. Alves, Jonatas Rodolfo Pereira dos Santos, Ariel da Silva Alsina
Date: 13 Feb. 2022
"""
import argparse
import logging
import tempfile
import pandas as pd
import numpy as np
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

    with tempfile.TemporaryDirectory() as tmpdir:
        abalone = pd.read_csv(artifact_path)


        logger.info("Creating artifact")

        # proper data pre-processing
        abalone['Age'] = abalone['Rings'].apply(lambda x: x + 1.5)
        abalone.drop('Rings', axis=1, inplace=True)

        abalone = abalone[abalone['Height'] != 0]  # need to drop these rows.

        corr = abalone.corr()
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        columns_to_drop = [column for column in upper_tri.columns if any(
            upper_tri[column] > 0.95)]

        abalone.drop(columns_to_drop, axis=1, inplace=True)

        abalone['Height'] = np.sqrt(abalone.loc[:, 'Height'])
        abalone['Length'] = np.sqrt(abalone.loc[:, 'Length'])

        abalone = pd.get_dummies(abalone, columns=['Sex'], prefix_sep='_')

        # data visualization for the dataframe with pandas
        profile = ProfileReport(
            abalone, title="Pandas Profiling Report", explorative=True)
        profile.to_file("abalone_profile.html")

        # memory persistence on the artifact
        abalone.to_csv("clean_data.csv")

        artifact = wandb.Artifact(
            name=args.artifact_name,
            type=args.artifact_type,
            description=args.artifact_description,
        )
        artifact.add_file("clean_data.csv")

        logger.info("Logging artifact")
        run.log_artifact(artifact)

        artifact.wait()


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
