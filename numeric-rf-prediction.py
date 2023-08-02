import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    try:
        data = pd.read_excel(file_path, engine='openpyxl')
        return data
    except Exception as e:
        print(f"Error occurred while loading data: {e}")
        return None

def preprocess_data(data):
    try:
        # Drop any rows with missing values
        data = data.dropna()

        # Convert non-numeric columns to numeric using label encoding
        label_encoder = LabelEncoder()
        for column in data.columns:
            if data[column].dtype == 'object':
                data[column] = label_encoder.fit_transform(data[column])

        return data
    except Exception as e:
        print(f"Error occurred during data preprocessing: {e}")
        return None

def train_model(train_data):
    try:
        # Separate features and target variable
        X = train_data.drop('drug_label', axis=1)
        y = train_data['drug_label']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model (you can use other classifiers as well)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")

        return model
    except Exception as e:
        print(f"Error occurred during model training: {e}")
        return None

def predict_labels(model, test_data):
    try:
        # Preprocess the test data
        test_data_processed = preprocess_data(test_data)

        if test_data_processed is None:
            return None

        # Make predictions using the trained model
        predictions = model.predict(test_data_processed)

        return predictions
    except Exception as e:
        print(f"Error occurred during prediction: {e}")
        return None

if __name__ == "__main__":
    # Load the training data
    training_data_path = "data.xlsx"
    training_data = load_data(training_data_path)

    # Load the test data
    test_data_path = "new_patient_data.xlsx"
    test_data = load_data(test_data_path)

    if training_data is not None and test_data is not None:
        # Preprocess the training data
        train_data_processed = preprocess_data(training_data)

        if train_data_processed is not None:
            # Train the model
            trained_model = train_model(train_data_processed)

            # Make predictions
            predictions = predict_labels(trained_model, test_data)

            if predictions is not None:
                # Print the predicted labels
                print("Predicted drug labels:")
                print(predictions)
            else:
                print("Prediction failed.")
        else:
            print("Training data preprocessing failed.")
    else:
        print("Failed to load training or test data.")
