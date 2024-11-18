# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib

# Load the dataset
data_path = 'content/bank-full.csv'
df = pd.read_csv(data_path, sep=';')
print("Dataset Loaded Successfully.")

print("\nFirst 5 Rows of Data:")
print(df.head())
print("\nData Info:")
print(df.info())

# Target variable encoding
df['y'] = df['y'].map({'yes': 1, 'no': 0})  # 'y' in binary

# Encode categorical features
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
label_encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_columns}
for col, encoder in label_encoders.items():
    df[col] = encoder.transform(df[col])

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Define features (X) and target (y)
X = df.drop(columns=['y'])
y = df['y']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"\nTraining set size: {X_train.shape}, Test set size: {X_test.shape}")

# Initialize and train the model
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

# Make predictions
y_pred = clf.predict(X_test)

# Print metrics
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model and label encoders
joblib.dump(clf, 'decision_tree_model.pkl')
for col, le in label_encoders.items():
    joblib.dump(le, f'label_encoder_{col}.pkl')

print("Model and encoders saved successfully.")
