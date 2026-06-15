import sys
import os
import pandas as pd

from src.logger import logging
from src.exception import MyException
from dataclasses import dataclass

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact, DataIngestionArtifact
from sklearn.metrics import f1_score
from src.constants import TARGET_COLUMN, SAVED_MODEL_NAME, SAVED_MODELS_DIR
from src.utils.main_utils import load_object, save_object

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float
    
class ModelEvaluation:
    
    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact : DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_atifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e
    
    def get_best_model(self):
        """ 
        Method Name    : get_best_model
        Description    : This function is used to get best model
        
        """
        try:
            model_path = self.model_eval_config.model_evaluation_after_best_model_file_path

            if not os.path.exists(model_path):
                return None

            return load_object(model_path)

        except Exception as e:
            raise MyException(e, sys) from e
        
    def _map_gender_columns(self, df):
        """ Map Gender column to 0 for Female and 1 for Male"""
        logging.info("Mapping 'Gender' column to binary values")
        df["Gender"] = df["Gender"].map({"Female" : 0, "Male" : 1}).astype(int)
        return df
    
    def _create_dummy_columns(self,df):
        """ Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical featres")
        df = pd.get_dummies(df, drop_first=True)
        return df
    
    def _rename_columns(self, df):
        """ Rename specific columns and ensure integer types for dummy columns."""
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns = {
                       "Vehicle_Age_< 1 Year" : "Vehicle_Age_lt_1_Year",
                       "Vehicle_Age_> 2 Years" : "Vehicle_Age_gt_2_Years"
                       })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df
    
    def _drop_id_columns(self, df):
        """ Drop the 'id' if it exists."""
        logging.info("Dropping 'id' column")
        if "_id" in df.columns:
            df = df.drop("_id", axis = 1)
        return df
    
    def evaluate_model(self) -> EvaluateModelResponse:
        """ 
        Method Name  : evaluate_model
        Description  : This function is used to evaluate trained model 
        with production model and choose best model
        
        Output       : Returns bool value based of validation results
        On Failure   : Write an exception log and than raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_atifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis = 1), test_df[TARGET_COLUMN]
            
            logging.info("Test data loaded and now transforming it for prediction")
            
            x = self._map_gender_columns(x)
            x = self._drop_id_columns(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)
            
            trained_model = load_object(file_path= self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"f1_score for this model: {trained_model_f1_score}")
            
            best_model_f1_score = None
            best_model = self.get_best_model()
            
            if best_model is not None:
                logging.info(f"Computing f1_score for production model")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"f1_score production Model: {best_model_f1_score}, f1_score New Trained Model: {trained_model_f1_score}")  
            
            tmp_best_model_score = 0  if best_model_f1_score is None else best_model_f1_score
            
            
            result = EvaluateModelResponse(trained_model_f1_score = trained_model_f1_score,
                                             best_model_f1_score=best_model_f1_score,
                                             is_model_accepted=trained_model_f1_score>tmp_best_model_score,
                                             difference=trained_model_f1_score-tmp_best_model_score
                                             )
            
            
            logging.info(f"Result: {result}")
            return result
        except Exception as e:
            raise MyException(e, sys) from e
        
    def intiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """ 
        Method Name   : intiate_model_evaluation
        Description   : This function is used to initiate all steps of the model evaluation
        
        Output        : Returns models evaluation artifact
        On Failure    : Write an exception log and then raise an exception
        """
        try:
            print("--------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            
            evaluate_model_response = self.evaluate_model()
            
            # Save trained model only if accepted
            if evaluate_model_response.is_model_accepted:
                trained_model = load_object(
                    self.model_trainer_artifact.trained_model_file_path
                )

                save_object(
                    self.model_eval_config.model_evaluation_after_best_model_file_path,
                    trained_model
                )
                
                # Save stable production model
                production_model_path = os.path.join(
                    SAVED_MODELS_DIR, SAVED_MODEL_NAME
                )

                save_object(
                    production_model_path,
                    trained_model
                )
                
                logging.info("New model accepted and saved as best model in both artifact and saved models folders.")

            else:
                logging.info("Existing model is better. New model rejected.")

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
                best_model_file_path=self.model_eval_config.model_evaluation_after_best_model_file_path
            )

            logging.info(f"Model Evaluation Artifact: {model_evaluation_artifact}")

            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e