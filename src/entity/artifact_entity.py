from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path : str
    test_file_path : str
    
@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_report_file_path: str
    
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    
@dataclass
class ClassificationMetricArtifact:
    f1_score : float
    precision_score : float
    recall_score : float
    accuracy_score : float

@dataclass
class ModelTrainerArtifact:
    models_path : dict
    models_metrics_report : dict[str, ClassificationMetricArtifact]
    models_f1_score_report : dict

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    changed_f1_score: float
    trained_best_model_path: str
    best_model_file_path : str