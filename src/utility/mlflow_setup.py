import dagshub
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
load_dotenv(override=True)  # Load environment variables from .env file
import os

# Local Environment MLflow Access
def setup_mlflow():
    """
    Initializes MLflow tracking for the project.
    """
    dagshub.init(repo_owner='oyebim', 
                repo_name='Fraudulent-Transaction-Detection', 
                mlflow=True)

# Set Experiment
mlflow.set_experiment('Fraudulent-Transaction-Detection_Models_For Finlora')


# Production Environment MLflow Access
def setup_mlflow():
    dagshub_token = os.getenv("MLFLOW_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("MLFLOW_TOKEN is not set")
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token  
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com/"
    repo_owner = "oyebim"
    repo_name = "Fraudulent-Transaction-Detection"

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
    # Set Experiment
    mlflow.set_experiment('Fraudulent-Transaction-Detection_Models_For Finlora')


