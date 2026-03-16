import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model():
    """
    This function loads the dataset, trains a RandomForestClassifier,
    and saves the model and scaler to disk.
    """
    try:
        # Step 1: Data ko file se padhkar 'df' variable mein daala jaata hai
        df = pd.read_csv('train.csv')
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: train.csv not found. Make sure the dataset is in the same directory.")
        return

    # Step 2: Hum check karte hain ki 'Id' column hai ya nahi
    if 'Id' in df.columns:
        # Agar hai, tabhi use drop karte hain
        df = df.drop('Id', axis=1)
    
    # Step 3: Check for the target column 'Cover_Type'
    if 'Cover_Type' not in df.columns:
        print("Error: Target column 'Cover_Type' not found in train.csv.")
        print("Please check your csv file to make sure it contains the 'Cover_Type' column.")
        return # Stop the script if the target is missing

    X = df.drop('Cover_Type', axis=1)
    y = df['Cover_Type']
    
    numerical_features = X.columns[:10]
    categorical_features = X.columns[10:]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled_num = X_train[numerical_features].copy()
    X_test_scaled_num = X_test[numerical_features].copy()
    
    X_train_scaled_num = scaler.fit_transform(X_train_scaled_num)
    X_test_scaled_num = scaler.transform(X_test_scaled_num)
    
    X_train_scaled_num = pd.DataFrame(X_train_scaled_num, columns=numerical_features, index=X_train.index)
    X_test_scaled_num = pd.DataFrame(X_test_scaled_num, columns=numerical_features, index=X_test.index)
    
    X_train_processed = pd.concat([X_train_scaled_num, X_train[categorical_features]], axis=1)
    X_test_processed = pd.concat([X_test_scaled_num, X_test[categorical_features]], axis=1)

    print("Training the RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_processed, y_train)
    print("Model training complete.")

    print("Evaluating the model...")
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
        
    joblib.dump(model, 'saved_models/model.joblib')
    joblib.dump(scaler, 'saved_models/scaler.joblib')
    print("\nModel and scaler have been saved successfully.")

if __name__ == '__main__':
    train_model()

