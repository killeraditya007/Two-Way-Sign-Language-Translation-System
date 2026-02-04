import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the Dataset
DATA_FILE = 'hand_signs_data.csv'
print("Loading dataset...")
data = pd.read_csv(DATA_FILE)

# 2. Separate Features (X) and Labels (y)
# X = All columns except 'label' (the coordinates)
# y = The 'label' column (the name of the sign)
X = data.drop('label', axis=1)
y = data['label']

# 3. Split Data into Training and Testing sets
# 80% used for training, 20% reserved to test accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
# RandomForest is great because it handles complex data well without much tuning
print("Training the model... (this might take a few seconds)")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Test the Model's Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the Model
# We save it as a '.p' file so we can load it later in our main app
joblib.dump(model, 'sign_language_model.p')
print("Model saved successfully as 'sign_language_model.p'!")