import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert data and labels to numpy arrays for better performance
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data into training and testing sets (80% training, 20% testing) with stratification
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create a Random Forest Classifier model
model = RandomForestClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Use the trained model to predict labels for the testing data
y_predict = model.predict(x_test)

# Calculate the accuracy of the model on the testing data
score = accuracy_score(y_predict, y_test)

# Print the accuracy as a percentage
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model to a pickle file for future use
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
