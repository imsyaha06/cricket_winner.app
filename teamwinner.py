# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np

# # Load the saved Random Forest model
# model = joblib.load('best_random_forest_model.pkl')

# # Load label encoders for transforming categorical input values
# label_encoders = joblib.load('label_encoders.pkl')  # Assuming you saved the encoders in a separate file

# # Title of the app
# st.title("ODI Match Winner Predictor")

# # Input fields for Team 1, Team 2, and Venue
# team1 = st.selectbox("Select Team 1:", label_encoders['team1'].classes_)
# team2 = st.selectbox("Select Team 2:", label_encoders['team2'].classes_)
# venue = st.selectbox("Select Venue:", label_encoders['venue'].classes_)

# # Predict button
# if st.button("Predict Winner"):
#     # Encode input values using saved label encoders
#     team1_encoded = label_encoders['team1'].transform([team1])[0]
#     team2_encoded = label_encoders['team2'].transform([team2])[0]
#     venue_encoded = label_encoders['venue'].transform([venue])[0]

#     # Create the feature array for the model
#     input_features = np.array([[team1_encoded, team2_encoded, venue_encoded]])

#     # Predict the winner using the Random Forest model
#     winner_encoded = model.predict(input_features)[0]

#     # Decode the winner back to the original team name
#     winner = label_encoders['winner'].inverse_transform([winner_encoded])[0]

#     # Display the prediction
#     st.write(f"The predicted winner is: **{winner}**")

############################################################################################################################################
# # # Create a Decision Tree Classifier
# # dt_model = DecisionTreeClassifier(random_state=42)

# # # Set hyperparameters to tune (these are examples, you can adjust based on your needs)
# # dt_params = {
# #     'max_depth': [3, 5, 10, None],  # Maximum depth of the tree
# #     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
# #     'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
# #     'criterion': ['gini', 'entropy']  # Function to measure the quality of a split
# # }

# # # Apply GridSearchCV to find the best hyperparameters for Decision Tree
# # grid_search_dt = GridSearchCV(dt_model, dt_params, cv=5, scoring='accuracy', n_jobs=-1)
# # grid_search_dt.fit(X_train, y_train)

# # # Get the best Decision Tree model based on GridSearchCV
# # best_dt_model = grid_search_dt.best_estimator_

# # # Predict on the test set
# # y_pred = best_dt_model.predict(X_test)

# # # Calculate accuracy
# # accuracy = accuracy_score(y_test, y_pred)
# # print(f"Best Decision Tree Hyperparameters: {grid_search_dt.best_params_}")
# # print(f"Decision Tree Accuracy: {accuracy:.4f}")
#########################################################################################################################################33
#####################1ST
# # Create a Random Forest Classifier
# rf_model = RandomForestClassifier(random_state=42)

# # Set hyperparameters to tune for Random Forest
# # Reduce hyperparameters for faster search
# rf_params = {
#     'n_estimators': [100, 200, 300],  # Number of trees in the forest
#     'max_depth': [5, 10, 20, None],   # Maximum depth of the tree
#     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
#     'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
#     'criterion': ['gini', 'entropy']  # Function to measure the quality of a split
# }

# # Apply GridSearchCV to find the best hyperparameters for Random Forest
# grid_search_rf = GridSearchCV(rf_model, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search_rf.fit(X_train, y_train)

# # Get the best Random Forest model based on GridSearchCV
# best_rf_model = grid_search_rf.best_estimator_

# # Predict on the test set using Random Forest
# y_pred_rf = best_rf_model.predict(X_test)

# # Calculate Random Forest accuracy
# rf_accuracy = accuracy_score(y_test, y_pred_rf)
# print(f"Best Random Forest Hyperparameters: {grid_search_rf.best_params_}")
# print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

 
# from sklearn.model_selection import RandomizedSearchCV
 
#######################2ND
# # Create a Random Forest Classifier
# rf_model = RandomForestClassifier(random_state=42)

# # Set hyperparameters to tune for Random Forest
# rf_params = {
#     'n_estimators': [100, 200, 300],  # Number of trees in the forest
#     'max_depth': [5, 10, 20, None],   # Maximum depth of the tree
#     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
#     'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
#     'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
#     'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider for the best split
# }

# # Apply RandomizedSearchCV to find the best hyperparameters for Random Forest
# random_search_rf = RandomizedSearchCV(rf_model, rf_params, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
# random_search_rf.fit(X_train, y_train)

# # Get the best Random Forest model based on RandomizedSearchCV
# best_rf_model = random_search_rf.best_estimator_

# # Predict on the test set using Random Forest
# y_pred_rf = best_rf_model.predict(X_test)

# # Calculate Random Forest accuracy
# rf_accuracy = accuracy_score(y_test, y_pred_rf)
# print(f"Best Random Forest Hyperparameters: {random_search_rf.best_params_}")
# print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
###########3RD################################################################
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import accuracy_score
# from joblib import dump
# import numpy as np

# # Assume X_train, y_train, X_test, and y_test are defined
# # Create a Random Forest Classifier
# rf_model = RandomForestClassifier(random_state=42)

# # Set hyperparameters to tune for Random Forest
# rf_params = {
#     'n_estimators': [50, 100],  # Reduced number of trees for smaller model size
#     'max_depth': [5, 10, 20, None],   # Maximum depth of the tree
#     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
#     'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
#     'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
#     'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider for the best split
# }

# # Apply RandomizedSearchCV to find the best hyperparameters for Random Forest
# random_search_rf = RandomizedSearchCV(rf_model, rf_params, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
# random_search_rf.fit(X_train, y_train)

# # Get the best Random Forest model based on RandomizedSearchCV
# best_rf_model = random_search_rf.best_estimator_

# # Predict on the test set using Random Forest
# y_pred_rf = best_rf_model.predict(X_test)

# # Calculate Random Forest accuracy
# rf_accuracy = accuracy_score(y_test, y_pred_rf)
# print(f"Best Random Forest Hyperparameters: {random_search_rf.best_params_}")
# print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# # Save the model with compression to reduce file size
# dump(best_rf_model, 'best_rf_model.joblib', compress=3)  # Adjust the compress level as needed
# ###########################################################6th#############################################
# import pickle
# # Create a Random Forest Classifier with fewer trees
# rf_model = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduce n_estimators for smaller size

# # Set hyperparameters to tune for Random Forest
# rf_params = {
#     'n_estimators': [100, 200, 300],  # Number of trees in the forest
#     'max_depth': [5, 10, 20, None],   # Maximum depth of the tree
#     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
#     'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
#     'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
#     'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider for the best split
# }

# # Apply RandomizedSearchCV to find the best hyperparameters for Random Forest
# random_search_rf = RandomizedSearchCV(rf_model, rf_params, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
# random_search_rf.fit(X_train, y_train)

# # Get the best Random Forest model based on RandomizedSearchCV
# best_rf_model = random_search_rf.best_estimator_

# # Predict on the test set using Random Forest
# y_pred_rf = best_rf_model.predict(X_test)

# # Calculate Random Forest accuracy
# rf_accuracy = accuracy_score(y_test, y_pred_rf)
# print(f"Best Random Forest Hyperparameters: {random_search_rf.best_params_}")
# print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# # Save the model using pickle with a lower protocol version to reduce size
# with open('random_forest_model.pkl', 'wb') as model_file:
#     pickle.dump(best_rf_model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

# # Alternatively, you can use joblib with a compressed option
# joblib.dump(best_rf_model, 'random_forest_model_compressed.joblib', compress=3)  # Use compression
#######################4TH##################################################################### 
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import accuracy_score
# from joblib import dump
# import numpy as np

# # Assume X_train, y_train, X_test, and y_test are defined
# # Create a Random Forest Classifier
# rf_model = RandomForestClassifier(random_state=42)

# # Set hyperparameters to tune for Random Forest
# rf_params = {
#     'n_estimators': [10, 20, 50],  # Reduced number of trees
#     'max_depth': [3, 5, 10],       # Further reduce max depth
#     'min_samples_split': [2, 5],   # Adjusted for fewer splits
#     'min_samples_leaf': [1, 2],    # Reduced leaf nodes
#     'criterion': ['gini', 'entropy'],
#     'max_features': ['sqrt']        # Limiting features considered
# }

# # Apply RandomizedSearchCV to find the best hyperparameters for Random Forest
# random_search_rf = RandomizedSearchCV(rf_model, rf_params, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
# random_search_rf.fit(X_train, y_train)

# # Get the best Random Forest model based on RandomizedSearchCV
# best_rf_model = random_search_rf.best_estimator_

# # Predict on the test set using Random Forest
# y_pred_rf = best_rf_model.predict(X_test)

# # Calculate Random Forest accuracy
# rf_accuracy = accuracy_score(y_test, y_pred_rf)
# print(f"Best Random Forest Hyperparameters: {random_search_rf.best_params_}")
# print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# # Save the model with higher compression to reduce file size
# dump(best_rf_model, 'best_rf_model.joblib', compress=5)  # Higher compression
###########################################5TH###################################################
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import accuracy_score
# import numpy as np
# import pickle
# import joblib

# # Create a Random Forest Classifier with fewer trees
# rf_model = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduce n_estimators for smaller size

# # Set hyperparameters to tune for Random Forest
# rf_params = {
#     'max_depth': [5, 10, 20],   # Maximum depth of the tree
#     'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
#     'min_samples_leaf': [1, 2],    # Minimum number of samples required to be at a leaf node
#     'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
#     'max_features': ['auto', 'sqrt']  # Number of features to consider for the best split
# }

# # Apply RandomizedSearchCV to find the best hyperparameters for Random Forest
# random_search_rf = RandomizedSearchCV(rf_model, rf_params, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
# random_search_rf.fit(X_train, y_train)

# # Get the best Random Forest model based on RandomizedSearchCV
# best_rf_model = random_search_rf.best_estimator_

# # Predict on the test set using Random Forest
# y_pred_rf = best_rf_model.predict(X_test)

# # Calculate Random Forest accuracy
# rf_accuracy = accuracy_score(y_test, y_pred_rf)
# print(f"Best Random Forest Hyperparameters: {random_search_rf.best_params_}")
# print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# # Save the model using pickle with a lower protocol version to reduce size
# with open('random_forest_model.pkl', 'wb') as model_file:
#     pickle.dump(best_rf_model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

# # Alternatively, you can use joblib with a compressed option
# joblib.dump(best_rf_model, 'random_forest_model_compressed.joblib', compress=3)  # Use compression
# # Save the best Random Forest model
# joblib.dump(best_rf_model, 'best_random_forest_model.pkl')
# print("Random Forest model saved as 'best_random_forest_model.pkl'")











































# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, classification_report
# import joblib
# from sklearn.model_selection import RandomizedSearchCV

# # Load the dataset
# data = pd.read_csv('ODI_Match_info.csv')



# # Remove columns where all values are empty strings or contain only whitespace
# data = data.replace(r'^\s*$', np.nan, regex=True)  # Replace empty strings or whitespace with NaN
# data = data.dropna(axis=1, how='all')  # Drop columns where all values are NaN

# print(data.head())  # Check a preview of the data


# # Separate features (X) and target (y)
# if 'winner' in data.columns:
#     y = data['winner']
#     X = data.drop('winner', axis=1)
# else:
#     print("The 'winner' column is not present in the dataset.")
#     y = None
#     X = data

# # Identify categorical columns
# categorical_cols = X.select_dtypes(include=['object']).columns

# # Apply OneHotEncoding to categorical columns
# ct = ColumnTransformer(transformers=[('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)], remainder='passthrough')
# X_encoded = ct.fit_transform(X)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# # Continue with model training and evaluation below

# # Apply filtering for training set
# train_mask = y_train.isin(y_train.value_counts()[y_train.value_counts() >= 5].index)
# X_train = X_train[train_mask]
# y_train = y_train[train_mask]

# # Apply filtering for testing set
# test_mask = y_test.isin(y_test.value_counts()[y_test.value_counts() >= 5].index)
# X_test = X_test[test_mask]
# y_test = y_test[test_mask]




# ############################################################################
# # # Create a Decision Tree Classifier
# # dt_model = DecisionTreeClassifier(random_state=42)

# # # Set more refined hyperparameters to tune for better performance
# # dt_params = {
# #     'max_depth': [5, 15, 20, None],  # Increased max_depth values
# #     'min_samples_split': [2, 10, 20],  # Increased minimum samples to split a node
# #     'min_samples_leaf': [1, 5, 10,],  # More samples per leaf node to prevent overfitting
# #     'criterion': ['gini', 'entropy'],  # Both Gini and Entropy to be tested
# #     'splitter': ['best', 'random']  # Add randomness to the split
# # }

# # # Apply GridSearchCV to find the best hyperparameters for Decision Tree
# # grid_search_dt = GridSearchCV(dt_model, dt_params, cv=5, scoring='accuracy', n_jobs=-1)  # Increased cv to 10
# # grid_search_dt.fit(X_train, y_train)

# # # Get the best Decision Tree model based on GridSearchCV
# # best_dt_model = grid_search_dt.best_estimator_

# # # Predict on the test set
# # y_pred = best_dt_model.predict(X_test)

# # # Calculate accuracy
# # accuracy = accuracy_score(y_test, y_pred)
# # print(f"Best Decision Tree Hyperparameters: {grid_search_dt.best_params_}")
# # print(f"Decision Tree Accuracy: {accuracy:.4f}")





# # ##################################################################################################3
# # import pickle
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.model_selection import RandomizedSearchCV
# # from sklearn.metrics import accuracy_score
# # import joblib  # Use joblib with compression for saving

# # # Create a Random Forest Classifier with fewer trees
# # rf_model = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduce n_estimators for smaller size

# # # Set hyperparameters to tune for Random Forest
# # rf_params = {
# #     'n_estimators': [50, 100],  # Fewer trees to reduce model size
# #     'max_depth': [5, 10, None],   # Maximum depth of the tree
# #     'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
# #     'min_samples_leaf': [1, 2],   # Minimum number of samples required to be at a leaf node
# #     'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
# #     'max_features': ['auto', 'sqrt']  # Number of features to consider for the best split
# # }

# # # Apply RandomizedSearchCV to find the best hyperparameters for Random Forest
# # random_search_rf = RandomizedSearchCV(rf_model, rf_params, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
# # random_search_rf.fit(X_train, y_train)

# # # Get the best Random Forest model based on RandomizedSearchCV
# # best_rf_model = random_search_rf.best_estimator_

# # # Predict on the test set using Random Forest
# # y_pred_rf = best_rf_model.predict(X_test)

# # # Calculate Random Forest accuracy
# # rf_accuracy = accuracy_score(y_test, y_pred_rf)
# # print(f"Best Random Forest Hyperparameters: {random_search_rf.best_params_}")
# # print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# # # Save the model using joblib with compression (higher compression rate, but slower)
# # joblib.dump(best_rf_model, 'best_rf_model_compressed.joblib', compress=3)  # Compress level 3 is a good balance

# # # Check the size of the saved model file
# # import os
# # file_size = os.path.getsize('best_rf_model_compressed.joblib') / (1024 * 1024)
# # print(f"Model file size: {file_size:.2f} MB")
# ############################################################################################################################################### 
 
# # import pickle
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.model_selection import RandomizedSearchCV
# # from sklearn.metrics import accuracy_score
# # import joblib  # Use joblib with compression for saving
# # import os

# # # Create a Random Forest Classifier with a slight increase in trees
# # rf_model = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True)  # Increased n_estimators to 100 for better accuracy

# # # Set hyperparameters to tune for Random Forest
# # rf_params = {
# #     'n_estimators': [75, 100, 150],  # Slight increase in the number of trees
# #     'max_depth': [10, 20, None],   # Increased depth to allow more complexity
# #     'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
# #     'min_samples_leaf': [1, 2],   # Minimum number of samples required to be at a leaf node
# #     'criterion': ['gini', 'entropy'],  # Gini and entropy to measure the quality of a split
# #     'max_features': ['auto', 'sqrt'],  # Number of features to consider for the best split
# #     'bootstrap': [True]  # Ensure bootstrapping to improve generalization
# # }

# # # Apply RandomizedSearchCV to find the best hyperparameters for Random Forest
# # random_search_rf = RandomizedSearchCV(rf_model, rf_params, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
# # random_search_rf.fit(X_train, y_train)

# # # Get the best Random Forest model based on RandomizedSearchCV
# # best_rf_model = random_search_rf.best_estimator_

# # # Predict on the test set using Random Forest
# # y_pred_rf = best_rf_model.predict(X_test)

# # # Calculate Random Forest accuracy
# # rf_accuracy = accuracy_score(y_test, y_pred_rf)
# # print(f"Best Random Forest Hyperparameters: {random_search_rf.best_params_}")
# # print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# # # Save the model using joblib with compression (higher compression rate, but slower)
# # joblib.dump(best_rf_model, 'best_rf_model_compressed.joblib', compress=3)  # Compress level 3 for balance between size and performance

# # # Check the size of the saved model file
# # file_size = os.path.getsize('best_rf_model_compressed.joblib') / (1024 * 1024)
# # print(f"Model file size: {file_size:.2f} MB")

# #############################################################################################################################################3
# # import pickle
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.model_selection import RandomizedSearchCV
# # from sklearn.metrics import accuracy_score
# # import joblib  # Use joblib with compression for saving
# # import os

# # # Create a Random Forest Classifier with a slight increase in trees and balanced class weights
# # rf_model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', bootstrap=False)  # Increase n_estimators to 150 for better accuracy

# # # Set hyperparameters to tune for Random Forest
# # rf_params = {
# #     'n_estimators': [100, 150],  # Slight increase in the number of trees to 150 for better accuracy
# #     'max_depth': [15, 20, None],   # Increased depth to allow more complexity
# #     'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
# #     'min_samples_leaf': [1, 2],   # Minimum number of samples required to be at a leaf node
# #     'criterion': ['gini', 'entropy'],  # Gini and entropy to measure the quality of a split
# #     'max_features': ['sqrt', 'log2'],  # Use sqrt or log2 for best split calculation
# #     'bootstrap': [False]  # Disable bootstrapping for potentially better splits and accuracy
# # }

# # # Apply RandomizedSearchCV to find the best hyperparameters for Random Forest
# # random_search_rf = RandomizedSearchCV(rf_model, rf_params, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
# # random_search_rf.fit(X_train, y_train)

# # # Get the best Random Forest model based on RandomizedSearchCV
# # best_rf_model = random_search_rf.best_estimator_

# # # Predict on the test set using Random Forest
# # y_pred_rf = best_rf_model.predict(X_test)

# # # Calculate Random Forest accuracy
# # rf_accuracy = accuracy_score(y_test, y_pred_rf)
# # print(f"Best Random Forest Hyperparameters: {random_search_rf.best_params_}")
# # print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# # # Save the model using joblib with compression (higher compression rate, but slower)
# # joblib.dump(best_rf_model, 'best_rf_model_compressed.joblib', compress=3)  # Compress level 3 for balance between size and performance

# # # Check the size of the saved model file
# # file_size = os.path.getsize('best_rf_model_compressed.joblib') / (1024 * 1024)
# # print(f"Model file size: {file_size:.2f} MB")

# #########################################################################################################################3
# import pickle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import accuracy_score
# import joblib  # Use joblib with compression for saving
# import os

# # Create a Random Forest Classifier with more trees and balanced class weights
# rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', oob_score=True, bootstrap=True)  # Increased n_estimators to 200 and enabled out-of-bag score for better accuracy

# # Set hyperparameters to tune for Random Forest
# rf_params = {
#     'n_estimators': [150, 200, 250],  # Slightly increased number of trees
#     'max_depth': [None],   # Allow trees to grow fully for better accuracy
#     'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
#     'min_samples_leaf': [1],   # Minimum number of samples required to be at a leaf node
#     'criterion': ['gini', 'entropy'],  # Gini and entropy to measure the quality of a split
#     'max_features': ['sqrt', 'log2'],  # sqrt or log2 for balanced feature selection
#     'bootstrap': [True]  # Use bootstrapping to improve model generalization
# }

# # Apply RandomizedSearchCV to find the best hyperparameters for Random Forest
# random_search_rf = RandomizedSearchCV(rf_model, rf_params, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
# random_search_rf.fit(X_train, y_train)

# # Get the best Random Forest model based on RandomizedSearchCV
# best_rf_model = random_search_rf.best_estimator_

# # Predict on the test set using Random Forest
# y_pred_rf = best_rf_model.predict(X_test)

# # Calculate Random Forest accuracy
# rf_accuracy = accuracy_score(y_test, y_pred_rf)
# print(f"Best Random Forest Hyperparameters: {random_search_rf.best_params_}")
# print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# # Save the model using joblib with compression (higher compression rate, but slower)
# joblib.dump(best_rf_model, 'best_rf_model_compressed.joblib', compress=3)  # Compress level 3 for balance between size and performance

# # Check the size of the saved model file
# file_size = os.path.getsize('best_rf_model_compressed.joblib') / (1024 * 1024)
# print(f"Model file size: {file_size:.2f} MB")

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

# # # Save the best Decision Tree model
# # joblib.dump(best_dt_model, 'best_decision_tree_model.pkl')
# # print("Decision Tree model saved as 'best_decision_tree_model.pkl'")

# # joblib.dump(ct, 'column_transformer.pkl')









































import streamlit as st
import pandas as pd
import joblib
import random

# Load the trained Decision Tree and Random Forest models and column transformer
dt_model = joblib.load('best_decision_tree_model.pkl')
rf_model = joblib.load('best_rf_model_compressed.joblib')  # Load the Random Forest model
ct = joblib.load('column_transformer.pkl')


teams = ['team1', 'team2']
# Randomly decide the toss winner
toss_winner = random.choice(teams)



# Function to make predictions
def predict_winner(team1, team2, venue):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'team1': [team1],
        'team2': [team2],
        'venue': [venue],
        'id': [0],  # Default values for missing columns
        'date': ['2023-01-01'],  # Placeholder date
        'dl_applied': [0],  # Assume D/L method is not applied
        'result': ['normal'],  # Placeholder result
        'toss_win': [toss_winner],  # Assume team1 won the toss
        'umpire3': [None],  # Assuming some columns might not be available
        'city': ['Unknown'],  # Placeholder city
        'player_of_match': [None],  # Placeholder
        'win_by_wickets': [0],  # Placeholder
        'umpire2': [None],  # Placeholder
        'toss_decision': ['bat'],  # Assume the toss decision is 'bat'
        'umpire1': [None],  # Placeholder
        'win_by_runs': [0],  # Placeholder
        'season': ['2023']  # Placeholder season
    })
    
    # Apply the column transformer to encode categorical features
    input_encoded = ct.transform(input_data)

    # Make predictions using both models
    dt_prediction = dt_model.predict(input_encoded)[0]
    rf_prediction = rf_model.predict(input_encoded)[0]

    return dt_prediction, rf_prediction

# Streamlit UI
st.title("ODI Match Winner Prediction")
st.write("Enter the details of the match to predict the winner:")

# User inputs
team1 = st.selectbox("Select Team 1", options=["Afghanistan", "Africa XI", "Asia XI", "Australia", "Bangladesh", "Bermuda", "Canada", 
                                               "England", "Hong Kong", "India", "Ireland", "Jersey", "Kenya", 
                                               "Namibia", "Nepal", "Netherlands", "New Zealand", "Oman", "Pakistan", 
                                               "Papua New Guinea", "Scotland", "South Africa", "Sri Lanka", "United Arab Emirates", 
                                               "United States of America", "West Indies", "Zimbabwe"])

team2 = st.selectbox("Select Team 2", options=["Afghanistan", "Africa XI", "Asia XI", "Australia", "Bangladesh", "Bermuda", "Canada", 
                                               "England", "Hong Kong", "India", "Ireland", "Jersey", "Kenya", 
                                               "Namibia", "Nepal", "Netherlands", "New Zealand", "Oman", "Pakistan", 
                                               "Papua New Guinea", "Scotland", "South Africa", "Sri Lanka", "United Arab Emirates", 
                                               "United States of America", "West Indies", "Zimbabwe"])   

venue = st.selectbox("Select Venue", options=["AMI Stadium", "Adelaide Oval", "Affies Park", "Al Amerat Cricket Ground Oman Cricket (Ministry Turf 1)",	"Al Amerat Cricket Ground Oman Cricket (Ministry Turf 2)", "Amini Park", "Port Moresby",
                                              "Andhra Cricket Association-Visakhapatnam District Cricket Association Stadium",	"Antigua Recreation Ground, St John's",	"Arbab Niaz Stadium", "Arnos Vale Ground",	"Arnos Vale Ground, Kingstown", 
                                              "Arnos Vale Ground, Kingstown, St Vincent", "Arun Jaitley Stadium", "Arun Jaitley Stadium, Delhi",	"Bangabandhu National Stadium",	"Bangabandhu National Stadium, Dhaka",	"Barabati Stadium",	"Barabati Stadium, Cuttack",
                                              "Barsapara Cricket Stadium",	"Barsapara Cricket Stadium, Guwahati",	"Basin Reserve", "Bay Oval", "Bay Oval, Mount Maunganui", "Beausejour Stadium, Gros Islet",	"Bellerive Oval",	"Bellerive Oval, Hobart", "Bert Sutcliffe Oval",	
                                              "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium", "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",	"Boland Bank Park, Paarl",	"Boland Park",	"Boland Park, Paarl",	"Brabourne Stadium",
                                              "Bready Cricket Club, Magheramason",	"Brian Lara Stadium, Tarouba, Trinidad", "Brisbane Cricket Ground",	"Brisbane Cricket Ground, Woolloongabba",	"Brisbane Cricket Ground, Woolloongabba, Brisbane",	"Buffalo Park",	"Buffalo Park", 
                                              "East London", "Bulawayo Athletic Club",	"Bundaberg Rum Stadium, Cairns",	"Cambusdoon New Ground",	"Cambusdoon New Ground, Ayr",	"Captain Roop Singh Stadium", "Captain Roop Singh Stadium, Gwalior", "Carisbrook",	"Castle Avenue", "Cazaly's Stadium, Cairns",
                                              "Central Broward Regional Park Stadium Turf Ground",	"Chevrolet Park",	"Chittagong Divisional Stadium",	"Choice Moosa Stadium, Pearland",	"City Oval, Pietermaritzburg",	"Civil Service Cricket Club, Stormont",	"Civil Service Cricket Club, Stormont, Belfast",
                                              "Clontarf Cricket Club Ground",	"Clontarf Cricket Club Ground, Dublin",	"Cobham Oval (New)", "County Ground", "County Ground, Bristol",	"County Ground, Chelmsford", "Daren Sammy National Cricket Stadium",
                                              "Daren Sammy National Cricket Stadium, Gros Islet",	"Darren Sammy National Cricket Stadium, Gros Islet",	"venue_Darren Sammy National Cricket Stadium, St Lucia",	"Davies Park, Queenstown",	
                                              "De Beers Diamond Oval",	"De Beers Diamond Oval, Kimberley",	"Diamond Oval",	"Diamond Oval, Kimberley",	"Docklands Stadium", "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",	"Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
                                              "Dubai International Cricket Stadium",	"Dubai Sports City Cricket Stadium", "Eden Gardens", "Eden Gardens, Kolkata",	"Eden Park", "Eden Park, Auckland",	"Edgbaston", "Edgbaston, Birmingham", "Feroz Shah Kotla",	"Gaddafi Stadium",	"Gaddafi Stadium, Lahore",
                                              "Galle International Stadium",	"Goodyear Park", "Goodyear Park, Bloemfontein",	"Grange Cricket Club Ground, Raeburn Place", "Grange Cricket Club Ground, Raeburn Place, Edinburgh",	"Grange Cricket Club, Raeburn Place",	"Greater Noida Sports Complex Ground",
                                              "Green Park", "Greenfield International Stadium",	"Greenfield International Stadium, Thiruvananthapuram",	"Gymkhana Club Ground",	"Gymkhana Club Ground, Nairobi",	"Hagley Oval",	"Hagley Oval, Christchurch", "Harare Sports Club",	"Hazelaarweg, Rotterdam", "Headingley",
                                               "Headingley, Leeds",	"Himachal Pradesh Cricket Association Stadium",	"Holkar Cricket Stadium", "Holkar Cricket Stadium, Indore", "ICC Academy",	"ICC Academy, Dubai",	"ICC Global Cricket Academy", "Indian Petrochemicals Corporation Limited Sports Complex Ground",
                                               "Iqbal Stadium",	"Iqbal Stadium, Faisalabad",	"JSCA International Stadium Complex",	"JSCA International Stadium Complex, Ranchi",	"Jade Stadium",	"Jade Stadium, Christchurch",	"Jaffery Sports Club Ground", "John Davies Oval",
                                               "Keenan Stadium",	"Kennington Oval",	"Kennington Oval, London",	"Kensington Oval, Barbados",	"Kensington Oval, Bridgetown", "Kensington Oval, Bridgetown, Barbados",	"Khan Shaheb Osman Ali Stadium", "Kingsmead", "Kingsmead, Durban",	
                                               "Kinrara Academy Oval",	"Lal Bahadur Shastri Stadium, Hyderabad, Deccan	venue_Lord's",	"Lord's, London",	"M Chinnaswamy Stadium", "M.Chinnaswamy Stadium",	"MA Aziz Stadium",	"MA Aziz Stadium, Chittagong",	"MA Chidambaram Stadium, Chepauk",
                                               "MA Chidambaram Stadium, Chepauk, Chennai",	"Madhavrao Scindia Cricket Ground",	"Maharani Usharaje Trust Cricket Ground",	"Maharashtra Cricket Association Stadium",	"Mahinda Rajapaksa International Cricket Stadium, Sooriyawewa",	"Mahinda Rajapaksa International Cricket Stadium, Sooriyawewa, Hambantota",
                                               "Malahide", "Mangaung Oval",	"Mangaung Oval, Bloemfontein",	"Mannofield Park",	"Mannofield Park, Aberdeen",	"Manuka Oval",	"Maple Leaf North-West Ground",	"Marrara Cricket Ground",	"Marrara Cricket Ground, Darwin", "McLean Park",	"McLean Park, Napier",	"Melbourne Cricket Ground",
                                               "Mission Road Ground, Mong Kok",	"Mombasa Sports Club Ground",	"Moosa Cricket Stadium, Pearland",	"Mulpani Cricket Ground",	"Multan Cricket Stadium",	"Nahar Singh Stadium",	"Nahar Singh Stadium, Faridabad",	"Narayanganj Osmani Stadium",	"Narendra Modi Stadium, Ahmedabad",	"National Cricket Stadium",	
                                               "National Cricket Stadium, Grenada",	"National Cricket Stadium, St George's", "National Stadium	venue_National Stadium, Karachi", "Nehru Stadium",	"Nehru Stadium, Fatorda",	"Nehru Stadium, Poona",	"New Wanderers Stadium",	"New Wanderers Stadium, Johannesburg",	"Newlands"	"Newlands, Cape Town"	"Niaz Stadium, Hyderabad",
                                               "North West Cricket Stadium, Potchefstroom",	"OUTsurance Oval",	"Old Hararians",	"Old Trafford",	"Old Trafford, Manchester",	"P Saravanamuttu Stadium",	"Pallekele International Cricket Stadium",	"Perth Stadium", "Providence Stadium",	"Providence Stadium, Guyana", "Punjab Cricket Association IS Bindra Stadium, Mohali",	
                                               "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",	"Punjab Cricket Association Stadium, Mohali",	"Queen's Park (New), St George's, Grenada",	"Queen's Park Oval", "Queen's Park Oval, Port of Spain", "Queen's Park Oval, Port of Spain, Trinidad",	"Queen's Park Oval, Trinidad", 
                                               "Queens Sports Club",	"Queens Sports Club, Bulawayo",	"Queenstown Events Centre",	"R Premadasa Stadium",	"R Premadasa Stadium, Colombo",	"R.Premadasa Stadium",	"R.Premadasa Stadium, Khettarama",	"Rajiv Gandhi International Cricket Stadium, Dehradun",	"Rajiv Gandhi International Stadium, Uppal",
                                               "Rajiv Gandhi International Stadium, Uppal, Hyderabad",	"Rangiri Dambulla International Stadium",	"Rawalpindi Cricket Stadium",	"Reliance Stadium",	"Riverside Ground",	"Riverside Ground, Chester-le-Street",	"Riverway Stadium, Townsville",	
                                               "Ruaraka Sports Club Ground"	"Sabina Park, Kingston",	"Sabina Park, Kingston, Jamaica", "Sardar Patel (Gujarat) Stadium, Motera",	"Sardar Patel Stadium, Motera",	"Saurashtra Cricket Association Stadium",	"Sawai Mansingh Stadium",	"Saxton Oval",	"Sector 16 Stadium",	"Seddon Park",	"Seddon Park, Hamilton", "Sedgars Park",	
                                               "Sedgars Park, Potchefstroom",	"Senwes Park",	"Senwes Park, Potchefstroom",	"Shaheed Chandu Stadium",	"Shaheed Veer Narayan Singh International Stadium, Raipur",	"Sharjah Cricket Association Stadium",	"Sharjah Cricket Stadium",	"Sheikh Abu Naser Stadium",	"Sheikh Zayed Stadium",	"Sheikhupura Stadium",	"Sher-e-Bangla National Cricket Stadium	",
                                               "Shere Bangla National Stadium",	"Shere Bangla National Stadium, Mirpur",	"Sinhalese Sports Club",	"Sinhalese Sports Club Ground",	"Sir Vivian Richards Stadium",	"Sir Vivian Richards Stadium, North Sound",	"Sophia Gardens",	"Sophia Gardens, Cardiff",	"Sportpark Het Schootsveld",	"Sportpark Maarschalkerweerd, Utrecht",	
                                               "St George's Park",	"St George's Park, Port Elizabeth",	"St Lawrence Ground",	"St Lawrence Ground, Canterbury",	"SuperSport Park",	"SuperSport Park, Centurion",	"Sydney Cricket Ground",	"Sylhet International Cricket Stadium",	"Takashinga Sports Club, Highfield, Harare",	"The Cooper Associates County Ground",	"The Rose Bowl",	"The Rose Bowl, Southampton	",
                                               "The Royal & Sun Alliance County Ground, Bristol",	"The Village, Malahide",	"The Village, Malahide, Dublin",	"The Wanderers Stadium",	"The Wanderers Stadium, Johannesburg",	"Titwood",	"Titwood, Glasgow",	"Tony Ireland Stadium, Townsville",	"Toronto Cricket, Skating and Curling Club",	"Trent Bridge",	"Trent Bridge, Nottingham",	
                                               "Tribhuvan University International Cricket Ground",	"Tribhuvan University International Cricket Ground, Kirtipur",	"United Cricket Club Ground, Windhoek",	"University Oval",	"VRA Cricket Ground",	"VRA Ground",	"VRA Ground, Amstelveen",	"Vidarbha C.A. Ground",	"Vidarbha Cricket Association Ground",	"Vidarbha Cricket Association Stadium, Jamtha",	"W.A.C.A. Ground",	
                                               "Wanderers Cricket Ground",	"Wanderers Cricket Ground, Windhoek",	"Wankhede Stadium",	"Wankhede Stadium, Mumbai",	"Warner Park, Basseterre",	"West End Park International Cricket Stadium, Doha",	"Western Australia Cricket Association Ground",	"Westpac Park, Hamilton",	"Westpac Stadium",	"Westpac Stadium, Wellington",	"Willowmoore Park",	"Willowmoore Park, Benoni",	
                                               "Windsor Park, Roseau",	"Zahur Ahmed Chowdhury Stadium",	"Zahur Ahmed Chowdhury Stadium, Chattogram",	"Zohur Ahmed Chowdhury Stadium",
])  # Replace with actual venues

# Prediction
if st.button("Predict Winner"):
    if team1 == team2:
        st.error("Team 1 and Team 2 cannot be the same!")
    else:
        dt_winner, rf_winner = predict_winner(team1, team2, venue)
        st.success(f"The predicted winner by Decision Tree is: {dt_winner}")
        st.success(f"The predicted winner by Random Forest is: {rf_winner}")
