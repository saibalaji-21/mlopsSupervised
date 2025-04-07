import os
import subprocess  # Added missing import
import sys
from mlflow.tracking import MlflowClient

# Define configurations
PORT = 1234  # Port for serving the model
HOST = "0.0.0.0"  # Host to serve on

def get_latest_run_id():
    """
    Fetches the latest run ID from MLflow tracking.
    Returns:
        str: The latest run ID
    Raises:
        ValueError: If no runs are found
    """
    client = MlflowClient()
    experiments = client.search_experiments()
    
    for experiment in experiments:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        if runs:
            return runs[0].info.run_id
    
    raise ValueError("No runs found in MLflow tracking.")

def serve_model(run_id):
    """
    Serves the model using the MLflow CLI.
    Args:
        run_id (str): The MLflow run ID containing the model
    Raises:
        RuntimeError: If model serving fails
    """
    model_path = f"runs:/{run_id}/model"
    print(f"\n🚀 Serving model from path: {model_path}")
    print(f"🌐 Endpoint will be available at: http://{HOST}:{PORT}/invocations\n")

    try:
        # Start the MLflow model server
        process = subprocess.Popen(
            [
                "mlflow", "models", "serve",
                "-m", model_path,
                "-p", str(PORT),
                "--host", HOST
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print real-time output
        for line in process.stdout:
            print(line, end='')
        
        # Wait for process to complete (with timeout)
        try:
            return_code = process.wait(timeout=30)
            if return_code != 0:
                raise RuntimeError(f"Model serving failed with code {return_code}")
        except subprocess.TimeoutExpired:
            print("Model server started successfully (process running in background)")
            
    except Exception as e:
        print(f"❌ Failed to serve the model: {str(e)}")
        if process:
            process.terminate()
        raise RuntimeError("Model serving failed") from e

if __name__ == "__main__":
    try:
        # Step 1: Fetch the latest run ID
        run_id = get_latest_run_id()
        print(f"🔍 Latest run ID: {run_id}")

        # Step 2: Serve the model
        serve_model(run_id)
        
    except Exception as e:
        print(f"🔥 Critical deployment error: {str(e)}")
        sys.exit(1)
