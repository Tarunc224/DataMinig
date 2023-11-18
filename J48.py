import javabridge
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random

# Start Java Virtual Machine
jvm.start()

# Load ARFF files
loader = Loader(classname="weka.core.converters.ArffLoader")
student_data = loader.load_file("student.arff")
employee_data = loader.load_file("employee.arff")

# Apply J48 algorithm to student dataset
j48_classifier_student = Classifier(classname="weka.classifiers.trees.J48")
j48_classifier_student.build_classifier(student_data)

# Apply J48 algorithm to employee dataset
j48_classifier_employee = Classifier(classname="weka.classifiers.trees.J48")
j48_classifier_employee.build_classifier(employee_data)

# Evaluate the model on student dataset
evaluation_student = Evaluation(student_data)
evaluation_student.crossvalidate_model(j48_classifier_student, student_data, 10, Random(1))
print("Student Dataset Evaluation Results:")
print(evaluation_student.summary())

# Evaluate the model on employee dataset
evaluation_employee = Evaluation(employee_data)
evaluation_employee.crossvalidate_model(j48_classifier_employee, employee_data, 10, Random(1))
print("\nEmployee Dataset Evaluation Results:")
print(evaluation_employee.summary())

# Stop Java Virtual Machine
jvm.stop()