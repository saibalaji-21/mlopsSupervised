import os
from databricks_cli.sdk.api_client import ApiClient
from databricks_cli.dbfs.api import DbfsApi

# Authenticate with Databricks using username and password
api_client = ApiClient(
    host=os.getenv("DATABRICKS_HOST"),          # Databricks workspace URL
    username=os.getenv("DATABRICKS_USERNAME"),  # Databricks username
    password=os.getenv("DATABRICKS_PASSWORD")   # Databricks password
)
dbfs_api = DbfsApi(api_client)

# Upload the model to DBFS
dbfs_path = "dbfs:/FileStore/models/model.pkl"  # Path in Databricks
local_path = "model.pkl"                        # Local path to the trained model

dbfs_api.put_file(local_path, dbfs_path, overwrite=True)
print(f"Model uploaded to {dbfs_path}")
