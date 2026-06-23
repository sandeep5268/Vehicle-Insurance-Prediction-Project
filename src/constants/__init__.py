import os
from datetime import date

# For MongoDB connection
DATABASE_NAME = "VehicleProject"
CONNECTION_NAME = "Vehicle-Project-Data"
MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = ""
ARTIFACT_DIR : str = "artifact"


TARGET_COLUMN = "Response"
CURRENT_YEAR = date.today().year
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME : str = "data.csv"
TRAIN_FILE_NAME : str = "train.csv"
TEST_FILE_NAME : str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

""" 
Data Ingestion related constants start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME = "Vehicle-Project-Data"
DATA_INGESTION_DIR_NAME : str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR : str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO : float = 0.25

""" 
Data Validation related constants start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME : str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

""" 
Data transformation related constants start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME : str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA : str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

""" 
Model Trainer related constants start with MODEL_TRAINER VAR NAME 
"""
MODEL_TRAINER_DIR_NAME :str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_METRIC_FILE_NAME: str = "trained_models_metrics.yaml"

""" 
RandomForestClassifier
"""
RFC_N_ESTIMATORS = 200
RFC_MIN_SAMPLES_SPLIT: int = 7
RFC_MIN_SAMPLE_LEAF : int = 6
RFC_MIN_SAMPLES_SPLIT_MAX_DEPTH : int = 10
RFC_MIN_SAMPLES_SPLIT_CRITERION: str ="entropy"
RFC_MIN_SAMPLES_SPLIT_RANDOM_STATE: int = 101

""" 
GradientBoostingClassifier
"""
GBC_MIN_SAMPLE_LEAF:int = 2
GBC_MIN_SAMPLES_SPLIT: int= 10
GBC_N_ESTIMATORS: int = 300
GBC_RANDOM_STATE: int = 42
GBC_SUBSAMPLE: float = 0.9

""" 
XGBoostClassifier
"""
XGB_N_ESTIMATOR: int = 300
XGB_MAX_DEPTH: int = 5
XGB_LEARNING_RATE: int = 0.05
XGB_COLSAMPLE_BYTREE: float =  0.8
XGB_GAMMA: float = 0.1
XGB_MIN_CHILD_WEIGHT: int = 5

""" 
Model Evaluation related constants
"""
MODEL_EVALUATON_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_DIR_NAME : str = "model_evaluation"
BEST_MODEL_DIR_NAME : str = "best_model"
MODEL_EVALUATION_AFTER_BEST_MODEL: str = "best_model.pkl"

""" 
Save models related constants
"""
SAVED_MODELS_DIR = "saved_models"
SAVED_MODEL_NAME = "best_model.pkl"

APP_HOST = "0.0.0.0"
APP_PORT = 5000