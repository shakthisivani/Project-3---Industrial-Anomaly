import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# File location
file_path = r'C:\Users\Ganesh Babu\Desktop\Final Project\Project 3 - Industrial Anomaly\sensor_data.csv'

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

# Preprocess the dataset
def preprocess_data(df):
    print("Original data shape:", df.shape)
    print("Columns with NaN values:", df.columns[df.isna().any()].tolist())
    print("NaN counts before preprocessing:")
    print(df.isna().sum())

    # Drop non-numeric columns like 'Timestamp' and 'Boiler Name'
    df = df.drop(columns=['Timestamp', 'Boiler Name'])
    
    # Handle NaN values - drop rows with NaN values in any column
    df = df.dropna()
    
    print("\nData shape after dropping NaN rows:", df.shape)
    print("NaN counts after preprocessing:")
    print(df.isna().sum())

    # Assuming the last column is the target variable (normal/anomaly)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print("\nUnique values in target variable:")
    print(y.value_counts())

    # Convert target to binary: 1 for normal, -1 for anomaly
    y = y.map({0: 1, 1: -1})
    
    # Check for any remaining NaN values in y
    if y.isna().any():
        print("Warning: NaN values found in target variable after mapping")
        print("NaN count in target:", y.isna().sum())
        # Fill NaN with a default value (e.g., 1 for 'normal')
        y = y.fillna(1)
    
    # Scale the feature set
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, df

# Train IsolationForest model
def train_anomaly_detection_model(X_train):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    # Predict the anomalies (-1 for anomaly, 1 for normal)
    y_pred = model.predict(X_test)

    # Ensure y_test is not empty
    if len(y_test) == 0:
        print("Error: y_test is empty after NaN handling.")
        return None

    # Ensure y_pred and y_test have consistent lengths
    if len(y_pred) != len(y_test):
        print(f"Error: Mismatch in lengths, y_pred: {len(y_pred)}, y_test: {len(y_test)}")
        return None
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=-1)
    
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    return y_pred

# Visualize the results
def visualize_results(df, y_pred, X_test):
    if y_pred is not None:
        # Create a new dataframe with only the test data
        df_test = pd.DataFrame(X_test, columns=['Temperature'])
        df_test['Predicted'] = y_pred
        print(df_test.head())

        # Plot the results
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df_test.index, y=df_test['Temperature'], hue=df_test['Predicted'], palette={1: 'green', -1: 'red'})
        plt.title("Anomaly Detection Results")
        plt.savefig('anomaly_detection_results.png')
        print("Results saved as 'anomaly_detection_results.png'")
        plt.close()

# Main workflow
print("Script started")

try:
    print("Loading data...")
    df = load_data(file_path)
    
    print("Preprocessing data...")
    X_scaled, y, scaler, df_processed = preprocess_data(df)

    # Check if there's any data left after preprocessing
    if len(X_scaled) == 0 or len(y) == 0:
        print("Error: No data left after preprocessing. Please check your data.")
    else:
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        print("Training model...")
        model = train_anomaly_detection_model(X_train)

        print("Evaluating model...")
        y_pred = evaluate_model(model, X_test, y_test)

        print("Visualizing results...")
        if y_pred is not None:
            visualize_results(df_processed, y_pred, X_test)

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print(traceback.format_exc())

print("Script finished")