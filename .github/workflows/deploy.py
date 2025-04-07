import os
import subprocess  # Added missing import
import sys
from mlflow.tracking import MlflowClient

# Define configurations
PORT = 1234  # Port for serving the model
HOST = "0.0.0.0"  # Make server accessible from outside

def get_latest_run_id():
    """
    Fetches the latest run ID from MLflow tracking.
    
    Returns:
        str: The run ID of the most recent experiment
    
    Raises:
        ValueError: If no runs are found in MLflow tracking
    """
    client = MlflowClient()
    experiments = client.search_experiments()  # Get all experiments
    
    # Search through all experiments to find the most recent run
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
        run_id (str): The MLflow run ID containing the model to serve
    
    Raises:
        RuntimeError: If model serving fails
    """
    model_path = f"runs:/{run_id}/model"
    print(f"\n=== Starting Model Serving ===")
    print(f"Model URI: {model_path}")
    print(f"Endpoint: http://{HOST}:{PORT}/invocations")
    
    try:
        # Start the MLflow model server
        process = subprocess.Popen(
            [
                "mlflow", "models", "serve",
                "-m", model_path,
                "-p", str(PORT),
                "--host", HOST,
                "--no-conda"  # Skip conda activation if not needed
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Print server output in real-time
        print("\nServer logs:")
        for line in process.stdout:
            print(line, end='')
            
        # Wait briefly to catch early failures
        try:
            return_code = process.wait(timeout=10)
            if return_code != 0:
                raise RuntimeError(f"Model server exited with code {return_code}")
        except subprocess.TimeoutExpired:
            print("\nModel server started successfully (running in background)")
            
    except Exception as e:
        print(f"\nERROR: Failed to start model server: {str(e)}")
        if process:
            process.terminate()
        raise RuntimeError("Model serving failed") from e

if __name__ == "__main__":
    try:
        # Step 1: Get the latest run
        run_id = get_latest_run_id()
        print(f"Latest run ID: {run_id}")
        
        # Step 2: Serve the model
        serve_model(run_id)
        
    except Exception as e:
        print(f"\nDEPLOYMENT FAILED: {str(e)}")
        sys.exit(1)
