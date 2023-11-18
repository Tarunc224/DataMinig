import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load ARFF file
data, meta = arff.loadarff(r"Student.arff")
# Convert ARFF data to DataFrame
df = pd.DataFrame(data)
# Encode categorical variables using LabelEncoder
categorical_columns = ["age", "income", "student", "credit-rating", "buyspc"]
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
# Split the dataset into features (X) and the target variable (y)
X = df.drop("buyspc", axis=1)
y = df["buyspc"]
print(X)
print(y)
