import os
import mlflow
import dagshub
from dotenv import load_dotenv

load_dotenv()

def setup_mlflow():
    repo_owner = os.getenv("DAGSHUB_USERNAME")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("DAGSHUB_TOKEN")
    tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
