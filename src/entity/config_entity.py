import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime   

TIMESTAMP : str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP
    
training_pipeline_config : TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir : str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path : str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME
    
@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)
    
@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA, TRAIN_FILE_NAME.replace("csv", "npy"))
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA, TEST_FILE_NAME.replace("csv", "npy"))
    transformed_object_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, PREPROCESSING_OBJECT_FILE_NAME)
    
@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_models_dir: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR)
    model_metric_file_path: str = os.path.join(trained_models_dir, MODEL_METRIC_FILE_NAME)
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    # RandomForestClassifier
    _n_estimators_rfc = RFC_N_ESTIMATORS
    _min_samples_split_rfc = RFC_MIN_SAMPLES_SPLIT
    _min_sample_leaf_rfc = RFC_MIN_SAMPLE_LEAF
    _max_depth_rfc = RFC_MIN_SAMPLES_SPLIT_MAX_DEPTH
    _criterion_rfc = RFC_MIN_SAMPLES_SPLIT_CRITERION
    _random_state_rfc = RFC_MIN_SAMPLES_SPLIT_RANDOM_STATE
    # GradientBoostingClassifier
    _min_sample_leaf_gbc = GBC_MIN_SAMPLE_LEAF
    _min_samples_split_gbc = GBC_MIN_SAMPLES_SPLIT
    _n_estimator_gbc = GBC_N_ESTIMATORS
    _random_state_gbc = GBC_RANDOM_STATE
    _subsample_gbc = GBC_SUBSAMPLE
    # XGBOOSTClassifier
    _n_estimator_xgb = XGB_N_ESTIMATOR
    _max_depth_xgb = XGB_MAX_DEPTH
    _learning_rate_xgb = XGB_LEARNING_RATE
    _colsample_bytree_xgb = XGB_COLSAMPLE_BYTREE
    _gamma_xgb = XGB_GAMMA
    _min_child_weight_xgb = XGB_MIN_CHILD_WEIGHT
    
        
@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATON_CHANGED_THRESHOLD_SCORE
    model_evaluation_dir_name: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR_NAME)
    model_evaluation_after_best_model_file_path: str = os.path.join(model_evaluation_dir_name, BEST_MODEL_DIR_NAME, MODEL_EVALUATION_AFTER_BEST_MODEL)
    
@dataclass
class VehiclePredictorConfig:
    model_file_path = os.path.join(
        "saved_models",
        "best_model.pkl"
    )