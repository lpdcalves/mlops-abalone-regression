"""
Creator: Ivanovitch Silva
Date: 30 Jan. 2022
Implement a machine pipeline component that
incorporate preprocessing and train stages.
"""
import argparse
import logging
import os

import yaml
import tempfile
import mlflow
from mlflow.models import infer_signature

import pandas as pd
import matplotlib.pyplot as plt
import wandb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor


# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()
    
def process_args(args):

    run = wandb.init(job_type="train")

    logger.info("Downloading and reading train artifact")
    local_path = run.use_artifact(args.train_data).file()
    df_train = pd.read_csv(local_path)

    # Spliting train.csv into train and validation dataset
    logger.info("Spliting data into train/val")
    # split-out train/validation and test dataset
    x_train, x_test, y_train, y_test = train_test_split(df_train.drop(labels=args.stratify,axis=1),
                                                      df_train[args.stratify],
                                                      test_size=args.val_size,
                                                      random_state=args.random_seed,
                                                      shuffle=True)
    
    logger.info("x train: {}".format(x_train.shape))
    logger.info("y train: {}".format(y_train.shape))
    logger.info("x val: {}".format(x_test.shape))
    logger.info("y val: {}".format(y_test.shape))

    sc = StandardScaler()
    sc.fit(x_train)

    X_train_std = sc.transform(x_train)
    X_test_std = sc.transform(x_test)
    X_train_std = pd.DataFrame(X_train_std, columns=x_train.columns)
    X_test_std = pd.DataFrame(X_test_std, columns=x_train.columns)

    X_train = X_train_std.values
    X_test = X_test_std.values
    y_train = y_train.values
    y_test = y_test.values

    # logger.info("Removal Outliers")
    # # temporary variable
    # x = x_train.select_dtypes("int64").copy()

    # # identify outlier in the dataset
    # lof = LocalOutlierFactor()
    # outlier = lof.fit_predict(x)
    # mask = outlier != -1

    # logger.info("x_train shape [original]: {}".format(x_train.shape))
    # logger.info("x_train shape [outlier removal]: {}".format(x_train.loc[mask,:].shape))

    # # dataset without outlier, note this step could be done during the preprocesing stage
    # x_train = x_train.loc[mask,:].copy()
    # y_train = y_train[mask].copy()

    # logger.info("Encoding Target Variable")
    # # define a categorical encoding for target variable
    # le = LabelEncoder()

    # # fit and transform y_train
    # y_train = le.fit_transform(y_train)

    # # transform y_test (avoiding data leakage)
    # y_val = le.transform(y_val)
    
    # logger.info("Classes [0, 1]: {}".format(le.inverse_transform([0, 1])))
    
    # # Pipeline generation
    # logger.info("Pipeline generation")
    
    # Get the configuration for the pipeline
    # with open(args.model_config) as fp:
    #     model_config = yaml.safe_load(fp)
        
    # Add it to the W&B configuration so the values for the hyperparams
    # are tracked
    # wandb.config.update(model_config)
    
    # The full pipeline 
    pipe = Pipeline([("Regressor", GradientBoostingRegressor(learning_rate=0.01, max_depth=4, min_samples_split=15, n_estimators=500))])

    # training 
    logger.info("Training")
    pipe.fit(x_train, y_train)

    # predict
    logger.info("Infering")
    predict = pipe.predict(x_test)
    
    # Evaluation Metrics
    logger.info("Evaluation metrics")
    
    # Uploading figures
    logger.info("Uploading figures")
    
    # Export if required
    if args.export_artifact != "null":
        export_model(run, pipe, x_test, predict, args.export_artifact)

        
def export_model(run, pipe, x_val, val_pred, export_artifact):

    # Infer the signature of the model
    signature = infer_signature(x_val, val_pred)

    with tempfile.TemporaryDirectory() as temp_dir:

        export_path = os.path.join(temp_dir, "model_export")

        mlflow.sklearn.save_model(
            pipe, # our pipeline
            export_path, # Path to a directory for the produced package
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature, # input and output schema
            input_example=x_val.iloc[:2], # the first few examples
        )

        artifact = wandb.Artifact(
            export_artifact,
            type="model_export",
            description="Decision Tree pipeline export",
        )
        
        # NOTE that we use .add_dir and not .add_file
        # because the export directory contains several
        # files
        artifact.add_dir(export_path)

        run.log_artifact(artifact)

        # Make sure the artifact is uploaded before the temp dir
        # gets deleted
        artifact.wait()        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Gradient Boosting Regressor",
        fromfile_prefix_chars="@",
    )
    
    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )

    parser.add_argument(
        "--export_artifact",
        type=str,
        help="Name of the artifact for the exported model. Use 'null' for no export.",
        required=False,
        default="null",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for the random number generator.",
        required=False,
        default=42
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size for the validation set as a fraction of the training set",
        required=False,
        default=0.3
    )

    parser.add_argument(
        "--stratify",
        type=str,
        help="Name of a column to be used for stratified sampling. Default: 'null', i.e., no stratification",
        required=False,
        default="null",
    )

    ARGS = parser.parse_args()

    process_args(ARGS)