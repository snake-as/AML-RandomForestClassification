I have created one script for drug prediction using as input  some important columns that I have insert into a train-set
After that I created one test-set without mentioning the expected column that we want to predicti. the  column ' drug_labels '

train-set:  [ 'LabId',	'ageAtDiagnosis',	'isRelapse'	,'isDenovo'	,'isTransformed', 'dxAtInclusion', 'specificDxAtInclusion',	'cumulativeTreatmentTypes',	'drug_label' ] 

test-set : [ 'LabId',	'ageAtDiagnosis',	'isRelapse'	,'isDenovo'	,'isTransformed', 'dxAtInclusion', 'specificDxAtInclusion',	'cumulativeTreatmentTypes' ] 

If anyone of  you want to add more columns that may he or she thinks that are more significal than these , simply edit the training and testing sets and add them into the code
I am attaching with name [ drug-prediction.py ] .

At the end of prediction I add a plot showing which drug is used the most
and after that , how many times exactly the drugs are used.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                           CODE IMPLEMENTATIONS DETAILS
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1. Import Required Libraries:
   - We import several libraries that are used throughout the code.
   - `pandas`: Used for data manipulation and loading Excel files.
   - `sklearn`: Scikit-learn library, used for machine learning tasks.
   - `LabelEncoder`: From sklearn, used for converting non-numeric data to numeric.

2. Define Helper Functions:
   - `load_data(file_path)`: This function loads the data from an Excel file specified by `file_path`.
   - `preprocess_data(data)`: This function preprocesses the data, including handling missing values and converting non-numeric columns to numeric using label encoding.
   - `train_model(train_data)`: This function trains a RandomForestClassifier on the provided training data.
   - `get_drug_label_mapping(data)`: This function extracts the unique drug names from the data and creates a mapping dictionary to map numeric labels back to drug names.
   - `predict_labels(model, test_data, drug_label_mapping)`: This function makes predictions on the test data using the trained model and converts numeric labels to drug names using the provided mapping.

3. Load Training and Test Data:
   - We define the paths for the training data ("data.xlsx") and test data ("new_patient_data.xlsx").
   - We use the `load_data` function to read the data from the Excel files into pandas DataFrames (`training_data` and `test_data`).

4. Preprocess the Data:
   - We use the `preprocess_data` function to preprocess the training data. It drops any rows with missing values and converts non-numeric columns to numeric using label encoding.
   - It is important to preprocess both the training and test data in the same way to ensure consistency.

5. Train the Model:
   - We use the `train_model` function to train a RandomForestClassifier on the preprocessed training data.
   - The model is trained on the features (X) and the target variable (y) obtained from the training data.
   - We split the training data into training and testing sets (80% training, 20% testing) to evaluate the model's performance.

6. Calculate Model Accuracy:
   - We calculate the accuracy of the model on the testing set using `accuracy_score` from sklearn.

7. Create Drug Label Mapping:
   - We use the `get_drug_label_mapping` function to create a mapping dictionary (`drug_label_mapping`) that maps numeric drug labels to their corresponding drug names.
   - This mapping is based on the unique drug names present in the training data.

8. Make Predictions:
   - We use the `predict_labels` function to make predictions on the test data using the trained model.
   - The predictions are initially in numeric format (0, 1, 2), representing the drug labels.
   - We use the `drug_label_mapping` to convert these numeric labels back to drug names.

9. Display Predicted Drug Labels:
   - Finally, we print the predicted drug labels along with their corresponding LabIds in a user-friendly format.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Throughout the code, we handle potential errors and print appropriate error messages to provide better feedback during data loading, preprocessing, model training, and prediction stages.

## The goal of this code is to load the training and test data, preprocess it, train a RandomForestClassifier on the training data, and then use this trained model to make predictions on the test data. The predictions are then converted back to the drug names for better interpretability.
