#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns



dataset = pd.read_csv('cardio_train.csv', delimiter = ';')

# drop the 'id' column from the dataframe
dataset = dataset.drop('id', axis=1)

dataset.info()

dataset.describe()
print(dataset.columns)

rcParams['figure.figsize'] = 20, 13
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
plt.colorbar()

dataset.hist()

print(dataset['cardio'].value_counts())

rcParams['figure.figsize'] = 8,6
plt.bar(dataset['cardio'].unique(), dataset['cardio'].value_counts(), color = ['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Cardio Classes')
plt.ylabel('Count')
plt.title('Count of each Cardio Class')

# One-hot encode categorical variables only
categorical_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
dataset = pd.get_dummies(dataset, columns=categorical_cols)

# Apply feature scaling to continuous variables only
continuous_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
ss = StandardScaler()
dataset[continuous_cols] = ss.fit_transform(dataset[continuous_cols])

y = dataset['cardio']
X = dataset.drop(['cardio'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

model = RandomForestClassifier(max_depth=6)
model.fit(X,y)

# Predicting the Test set results
y_pred = model.predict(X_test)

# make confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.show()

# Calculate accuracy score
acc = accuracy_score(y_test, y_pred)
print (f'RF acc score: {acc:.2f} ')

new_data = [
  {
    'age': 19345,
    'gender': 1,
    'height': 184,
    'weight': 95.0,
    'ap_hi': 150,
    'ap_lo': 80,
    'cholesterol': 2,
    'gluc': 2,
    'smoke': 1,
    'alco': 1,
    'active': 0,
  },
  {
    'age': 12345,
    'gender': 2,
    'height': 161,
    'weight': 52.0,
    'ap_hi': 130,
    'ap_lo': 70,
    'cholesterol': 1,
    'gluc': 1,
    'smoke': 0,
    'alco': 0,
    'active': 0,
  },
  {
    'age': 16345,
    'gender': 2,
    'height': 161,
    'weight': 52.0,
    'ap_hi': 130,
    'ap_lo': 70,
    'cholesterol': 3,
    'gluc': 3,
    'smoke': 0,
    'alco': 0,
    'active': 1,
  }
]

print()

new_data = pd.DataFrame(new_data)

new_data = pd.get_dummies(new_data, columns=categorical_cols)
new_data[continuous_cols] = ss.fit_transform(new_data[continuous_cols])


# Predict with new data and create dataframe
new_y = pd.DataFrame(model.predict(new_data))

# apply species information based on the prediction
new_y[1] = new_y[0].apply(lambda x: 'no heartdisease' if x == 0 else 'heart disease')


print(new_y)