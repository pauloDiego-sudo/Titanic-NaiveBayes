from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import joblib

# Load the Titanic dataset uploaded by the user
file_path = 'train.csv'
titanic_data = pd.read_csv(file_path)

# Preprocessing the dataset

# Dropping unnecessary columns
titanic_data_clean = titanic_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Handling missing values
# Fill missing values in "Age" with the median age
titanic_data_clean["Age"] = titanic_data_clean["Age"].fillna(titanic_data_clean["Age"].median())

# Fill missing values in "Embarked" with the mode
titanic_data_clean["Embarked"] = titanic_data_clean["Embarked"].fillna(titanic_data_clean["Embarked"].mode()[0])

# Convert categorical variables like "Sex" and "Embarked" into numerical values
label_encoder_sex = LabelEncoder()
label_encoder_embarked = LabelEncoder()

titanic_data_clean["Sex"] = label_encoder_sex.fit_transform(titanic_data_clean["Sex"])
titanic_data_clean["Embarked"] = label_encoder_embarked.fit_transform(titanic_data_clean["Embarked"])

# Save the LabelEncoder objects
joblib.dump(label_encoder_sex, 'label_encoder_sex.pkl')
joblib.dump(label_encoder_embarked, 'label_encoder_embarked.pkl')

# Splitting the data into features (X) and target (y)
X = titanic_data_clean.drop(columns=["Survived"])
y = titanic_data_clean["Survived"]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Saving the model using joblib
joblib.dump(nb_model, 'titanic_naive_bayes_model.pkl')

# Making predictions on the test set
y_pred = nb_model.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy}")
