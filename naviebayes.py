import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset into a Pandas DataFrame
data = pd.read_csv('employee.csv')

# Encode the categorical attributes (age, salary, performance) to numeric values
le = LabelEncoder()
data['age'] = le.fit_transform(data['age'])
data['salary'] = le.fit_transform(data['salary'])
data['performance'] = le.fit_transform(data['performance'])

# Split the dataset into features (X) and target (y)
X = data[['age', 'salary']]
y = data['performance']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)