import os
from mlflow.tracking import MlflowClient

# Define configurations
PORT = 1234  # Port for serving the model

def get_latest_run_id():
    """
    Fetches the latest run ID from MLflow tracking.
    """
    client = MlflowClient()
    experiments = client.search_experiments()  # Get all experiments
    latest_run = None

    for experiment in experiments:
        # Search for the latest run in each experiment
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],  # Order by start time (newest first)
            max_results=1  # Only fetch the latest run
        )
        if runs:
            latest_run = runs[0]  # Get the first (latest) run
            break

    if not latest_run:
        raise ValueError("No runs found in MLflow tracking.")

    return latest_run.info.run_id

def serve_model(run_id):
    """
    Serves the model using the MLflow CLI.
    """
    model_path = f"runs:/{run_id}/model"  # Construct the model path
    print(f"Serving model from path: {model_path}")

    try:
        # Serve the model using the MLflow CLI
        subprocess.run(
            ["mlflow", "models", "serve", "-m", model_path, "-p", str(PORT)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to serve the model: {e}")
        raise

if __name__ == "__main__":
    try:
        # Step 1: Fetch the latest run ID
        run_id = get_latest_run_id()
        print(f"Latest run ID: {run_id}")

        # Step 2: Serve the model
        serve_model(run_id)
    except Exception as e:
        print(f"Error during deployment: {e}")
        raise
