import sys
import os
import numpy as np
from typing import Tuple
from dataclasses import asdict
from src.logger import logging
from src.exception import MyException

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from src.utils.main_utils import load_numpy_array_data, save_object, write_yaml_file
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """ 
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        
    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[dict, dict, dict]:
        """ 
        Method Name    :  get_model_object_and_report
        Description    :  This function trains multiple models with specified parameters
        
        Output         :  Returns metric artifact object and trained model object
        On Failure     :  Write an exception log and then raise an exception
        """
        try:
            logging.info("Training Multiple Models with specified parameters")
            
            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train[:,:-1], train[:,-1], test[:,:-1], test[:, -1]
            logging.info("train-test split done.")
            
            # Initialize Multiple Models with specified parameters
            models = {
                "RandomForest" : RandomForestClassifier(
                    n_estimators= self.model_trainer_config._n_estimators_rfc,
                    min_samples_split= self.model_trainer_config._min_samples_split_rfc,
                    min_samples_leaf= self.model_trainer_config._min_sample_leaf_rfc,
                    max_depth= self.model_trainer_config._max_depth_rfc,
                    criterion= self.model_trainer_config._criterion_rfc,
                    random_state= self.model_trainer_config._random_state_rfc
                ),
                "GradientBoosting" : GradientBoostingClassifier(
                    min_samples_leaf= self.model_trainer_config._min_sample_leaf_gbc,
                    min_samples_split= self.model_trainer_config._min_samples_split_gbc,
                    n_estimators= self.model_trainer_config._n_estimator_gbc,
                    random_state= self.model_trainer_config._random_state_gbc,
                    subsample= self.model_trainer_config._subsample_gbc
                ),
                "XGBoost" : XGBClassifier(
                    n_estimators = self.model_trainer_config._n_estimator_xgb,
                    max_depth = self.model_trainer_config._max_depth_xgb,
                    learning_rate = self.model_trainer_config._learning_rate_xgb,
                    colsample_bytree = self.model_trainer_config._colsample_bytree_xgb,
                    gamma = self.model_trainer_config._gamma_xgb,
                    min_child_weight = self.model_trainer_config._min_child_weight_xgb
                )
                
            }
            
            # Fit the model
            logging.info("Model training going on....!")
            models_f1_score_report = {}
            models_metrics_report = {}
            model_paths = {}
            for model_name, model in models.items():
                logging.info(f"{model_name} Model training started..!")
                model.fit(x_train, y_train)
                logging.info(f"{model_name} Model training done.")

                model_path = os.path.join(
                    self.model_trainer_config.trained_models_dir,f"{model_name}.pkl"
                )
                save_object(
                    file_path=model_path,
                    obj=model
                )
                
                # Predictions and evaluation metrics
                y_pred = model.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                
                # Storing the metrics and model paths in respective dictionaries
                models_f1_score_report[model_name] = f1
                metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall, accuracy_score=accuracy)
                models_metrics_report[model_name] = asdict(metric_artifact)
                model_paths[model_name] = model_path
                
            return models_metrics_report, models_f1_score_report, model_paths
        except Exception as e:
            raise MyException(e, sys) from e
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """ 
        Method Name    :  initiate_model_trainer 
        Description    :  This function initiates the model training steps
        
        Output         :  Returns model trainer artifact
        On Failure     :  Write an exception log and then raise an exception
        """
        try:
            print("-------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")
            
            # Train Model and get metrics
            logging.info("Models Training Started...!")
            models_metrics_report, models_f1_score_report, model_paths = self.get_model_object_and_report(train = train_arr, test = test_arr)
            logging.info("Models Training completed successfully..!")
            
            logging.info("Models metrics report storing as started .....")
            # Store the models metrics report as yaml file
            write_yaml_file(
                self.model_trainer_config.model_metric_file_path,
                content = models_metrics_report
            )
            logging.info("Models metrics report is stored as yaml file")
            
            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                models_path = model_paths,
                models_metrics_report= models_metrics_report,
                models_f1_score_report = models_f1_score_report
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e