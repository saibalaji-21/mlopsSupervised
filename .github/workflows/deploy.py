import os
import requests
import base64

# Authenticate with Databricks using username and password
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_USERNAME = os.getenv("DATABRICKS_USERNAME")
DATABRICKS_PASSWORD = os.getenv("DATABRICKS_PASSWORD")

# Debugging: Print environment variables
print(f"DATABRICKS_HOST: {DATABRICKS_HOST}")
print(f"DATABRICKS_USERNAME: {'***' if DATABRICKS_USERNAME else 'Not set'}")
print(f"DATABRICKS_PASSWORD: {'***' if DATABRICKS_PASSWORD else 'Not set'}")

# Ensure DATABRICKS_HOST includes the 'https://' scheme
if not DATABRICKS_HOST:
    raise ValueError("DATABRICKS_HOST is not set or is empty.")
if not DATABRICKS_HOST.startswith("https://"):
    DATABRICKS_HOST = f"https://{DATABRICKS_HOST}"

# Get an authentication token using username and password
auth_url = f"{DATABRICKS_HOST}/api/2.0/token/create"
auth_data = {
    "lifetime_seconds": 3600,  # Token expires in 1 hour
    "comment": "Temporary token for CI/CD pipeline"
}
response = requests.post(auth_url, auth=(DATABRICKS_USERNAME, DATABRICKS_PASSWORD), json=auth_data)
if response.status_code != 200:
    print(f"Response status code: {response.status_code}")
    print(f"Response body: {response.text}")
    raise Exception(f"Failed to authenticate with Databricks: {response.text}")

token = response.json().get("token_value")
if not token:
    raise Exception("Failed to retrieve token from Databricks.")

# Upload the model to DBFS using the REST API
dbfs_url = f"{DATABRICKS_HOST}/api/2.0/dbfs/put"
dbfs_path = "dbfs:/FileStore/models/model.pkl"  # Path in Databricks
local_path = "model.pkl"  # Local path to the trained model

with open(local_path, "rb") as file:
    file_content = base64.b64encode(file.read()).decode("utf-8")

response = requests.post(
    dbfs_url,
    headers={"Authorization": f"Bearer {token}"},
    json={
        "path": dbfs_path,
        "contents": file_content,
        "overwrite": True
    }
)

if response.status_code == 200:
    print(f"Model uploaded to {dbfs_path}")
else:
    print(f"Response status code: {response.status_code}")
    print(f"Response body: {response.text}")
    raise Exception(f"Failed to upload model to DBFS: {response.text}")
