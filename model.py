import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

from flask import Flask, request, render_template

# Load the dataset
df = pd.read_csv("celiac_disease_lab_data_final.csv")

# Drop any rows with missing values
df.dropna(inplace=True)

# Define features and target variable
features = ['Age', 'Abdominal', 'Short_Stature', 'Weight_loss', 'IgA', 'IgG']
target = 'Disease_Diagnose'  # Target column

X = df[features]
y = df[target]

# Encode categorical features
le = LabelEncoder()
X_encoded = X.copy()
for col in X.columns:
    if X[col].dtype == 'object':
        X_encoded[col] = le.fit_transform(X[col])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Initialize and fit Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Save the model to disk
pickle.dump(rf_classifier, open('model.pkl', 'wb'))

# Flask app initialization
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = int(request.form['age'])
    abdominal = int(request.form['abdominal'])
    short_stature = int(request.form['shortStature'])
    weight_loss = int(request.form['weightLoss'])
    iga = float(request.form['iga'])
    igg = float(request.form['igg'])

    # Make prediction
    input_data = [[age, abdominal, short_stature, weight_loss, iga, igg]]
    prediction = rf_classifier.predict(input_data)

    # Return prediction
    return str(prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
