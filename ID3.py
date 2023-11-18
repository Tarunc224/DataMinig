import csv
import math

# Function to calculate entropy
def entropy(data, target_attribute):
    # Count the occurrences of each target value
    target_counts = {}
    for row in data:
        target_value = row[target_attribute]
        if target_value not in target_counts:
            target_counts[target_value] = 0
        target_counts[target_value] += 1

    # Calculate entropy using the formula
    entropy_value = 0
    total_instances = len(data)
    for target_value in target_counts:
        probability = target_counts[target_value] / total_instances
        entropy_value -= probability * math.log2(probability)

    return entropy_value

# Function to calculate information gain for an attribute
def information_gain(data, attribute, target_attribute):
    total_entropy = entropy(data, target_attribute)
    total_instances = len(data)

    attribute_values = set([row[attribute] for row in data])
    weighted_entropy = 0

    for value in attribute_values:
        subset = [row for row in data if row[attribute] == value]
        subset_entropy = entropy(subset, target_attribute)
        probability = len(subset) / total_instances
        weighted_entropy += probability * subset_entropy

    return total_entropy - weighted_entropy

# Function to choose the best attribute to split on
def choose_best_attribute(data, attributes, target_attribute):
    best_attribute = None
    max_info_gain = -1

    for attribute in attributes:
        info_gain = information_gain(data, attribute, target_attribute)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_attribute = attribute

    return best_attribute

# Recursive ID3 algorithm to build the decision tree
def id3(data, attributes, target_attribute):
    target_values = set([row[target_attribute] for row in data])

    # If all instances have the same target value, return that value
    if len(target_values) == 1:
        return target_values.pop()

    # If there are no attributes left to split on, return the majority target value
    if len(attributes) == 0:
        majority_target = max(set([row[target_attribute] for row in data]), key=[row[target_attribute] for row in data].count)
        return majority_target

    # Choose the best attribute to split on
    best_attribute = choose_best_attribute(data, attributes, target_attribute)

    # Create a new decision tree node with the best attribute as its label
    tree = {best_attribute: {}}
    attribute_values = set([row[best_attribute] for row in data])

    # Recursively build the subtree for each attribute value
    for value in attribute_values:
        subset = [row for row in data if row[best_attribute] == value]
        subtree = id3(subset, [attr for attr in attributes if attr != best_attribute], target_attribute)
        tree[best_attribute][value] = subtree

    return tree

# Read data from a CSV file
data = []
with open(r"employee.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)

# List of attributes (excluding the target attribute)
attributes = ['age', 'salary']

# Target attribute
target_attribute = 'performance'

# Build the decision tree
decision_tree = id3(data, attributes, target_attribute)

# Print the resulting decision tree
import pprint
pprint.pprint(decision_tree)