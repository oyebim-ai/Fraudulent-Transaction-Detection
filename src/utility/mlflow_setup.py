import dagshub
import mlflow
import mlflow.sklearn


def setup_mlflow():
    """
    Initializes MLflow tracking for the project.
    """
    dagshub.init(repo_owner='oyebim', 
                repo_name='Fraudulent-Transaction-Detection', 
                mlflow=True)

# Set Experiment
mlflow.set_experiment('Fraudulent-Transaction-Detection_Models_For Finlora')