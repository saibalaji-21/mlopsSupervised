import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import joblib

# Load dataset
data = pd.read_csv('winequality-red.csv', delimiter=';')

# Drop rows with missing values
data = data.dropna()

# Set MLflow experiment
mlflow.set_experiment('wine-quality-prediction')

# Elastic Net takes two parameters: alpha (ridge) and l1_ratio (lasso)
def train_model(alpha, l1_ratio):
    # Develop train-test data
    train, test = train_test_split(data, test_size=0.3, random_state=1234)
    train_x = train.drop(['quality'], axis=1)  # Dropping the target column 'quality'
    test_x = test.drop(['quality'], axis=1)
    train_y = train[['quality']]
    test_y = test[['quality']]

    # Start MLflow run
    with mlflow.start_run(run_name='wine-quality-regression') as run:
        # Model building
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        lr.fit(train_x, train_y)

        # Model prediction
        predicted_data = lr.predict(test_x)

        # Model evaluation
        rmse = np.sqrt(mean_squared_error(test_y, predicted_data))
        mae = mean_absolute_error(test_y, predicted_data)
        r2 = r2_score(test_y, predicted_data)

        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R2:", r2)

        # Log parameters, metrics, and model
        mlflow.log_param('alpha', alpha)
        mlflow.log_param('l1_ratio', l1_ratio)

        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('MAE', mae)
        mlflow.log_metric('R2', r2)

        mlflow.sklearn.log_model(lr, 'model')  # Log the model as an artifact

        # Save the trained model locally
        joblib.dump(lr, 'model.pkl')

# Train the model with specific hyperparameters
if __name__ == "__main__":
    train_model(0.3, 0.6)
