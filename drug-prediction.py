import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    try:
        data = pd.read_excel(file_path, engine='openpyxl')
        return data
    except Exception as e:
        print(f"Error occurred while loading data: {e}")
        return None

def preprocess_data(data):
    try:
        data = data.dropna()

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
        X = train_data.drop('drug_label', axis=1)
        y = train_data['drug_label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")

        return model
    except Exception as e:
        print(f"Error occurred during model training: {e}")
        return None

def get_drug_label_mapping(data):
    try:
        label_encoder = LabelEncoder()
        unique_drugs = data['drug_label'].unique()
        numeric_labels = label_encoder.fit_transform(unique_drugs)

        drug_label_mapping = dict(zip(numeric_labels, unique_drugs))

        return drug_label_mapping
    except Exception as e:
        print(f"Error occurred during mapping creation: {e}")
        return None

def predict_labels(model, test_data, drug_label_mapping):
    try:
        test_data_processed = preprocess_data(test_data)

        if test_data_processed is None:
            return None

        predictions = model.predict(test_data_processed)

        drug_names = [drug_label_mapping[label] for label in predictions]

        return drug_names
    except Exception as e:
        print(f"Error occurred during prediction: {e}")
        return None

if __name__ == "__main__":
    training_data_path = "train-set.xlsx" # The training data-set as is saved on my file
    training_data = load_data(training_data_path)

    test_data_path = "test-set.xlsx" # The test data-set as is saved on my file
    test_data = load_data(test_data_path) 

    if training_data is not None and test_data is not None:
        drug_label_mapping = get_drug_label_mapping(training_data)

        if drug_label_mapping is not None:
            train_data_processed = preprocess_data(training_data)

            if train_data_processed is not None:
                trained_model = train_model(train_data_processed)

                predictions = predict_labels(trained_model, test_data, drug_label_mapping)

                if predictions is not None:
                    print("Predicted drug labels:")
                    for lab_id, drug in zip(test_data['LabId'], predictions):
                        print(f"{lab_id}: {drug}")
                else:
                    print("Prediction failed.")

                # Create a bar plot to show drug usage in the training data
                drug_usage = training_data['drug_label'].value_counts().reset_index()
                drug_usage.columns = ['Drug', 'Usage']
                plt.figure(figsize=(8, 6))
                sns.barplot(x='Drug', y='Usage', data=drug_usage, palette='Blues')
                plt.title('Drug Usage in Training Data')
                plt.xlabel('Drug')
                plt.ylabel('Frequency')
                plt.show()

                # Show the count of drug usage
                print("\nDrug usage count:")
                print(drug_usage)

            else:
                print("Training data preprocessing failed.")
        else:
            print("Failed to create drug label mapping.")
    else:
        print("Failed to load training or test data.")
