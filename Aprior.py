import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load the dataset
data = pd.read_csv(r"super market.csv")

# Convert 'y' and 'n' to boolean values (True and False)
data = data.applymap(lambda x: True if x == 'y' else False)

# Perform one-hot encoding
one_hot_encoded = pd.get_dummies(data)

# Find frequent item sets with minimum support
frequent_itemsets = apriori(one_hot_encoded, min_support=0.2, use_colnames=True)

# Generate association rules with minimum confidence and compute lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Display the association rules
print("Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Save the rules to a CSV file if needed
# rules.to_csv("association_rules.csv", index=False)