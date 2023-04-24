#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict


warnings.filterwarnings('ignore')

dataset = pd.read_csv('cardio_train.csv', delimiter = ';')

# drop the 'id' column from the dataframe
dataset = dataset.drop('id', axis=1)

rcParams['figure.figsize'] = 20, 13
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
plt.colorbar()

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


rf_scores = []
estimators = [10, 100, 200, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(rf_classifier.score(X_test, y_test))


colors = rainbow(np.linspace(0, 1, len(estimators)))
plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)
for i in range(len(estimators)):
    plt.text(i, rf_scores[i], rf_scores[i])
plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])
plt.xlabel('Number of estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different number of estimators')


print("The score for Random Forest Classifier is {}% with {} estimators.".format(rf_scores[1]*100, [100, 500]))

y_pred = cross_val_predict(rf_classifier, X, y, cv=5, method="predict_proba")[:, 1]

print("Precision Score: ", precision_score(y, y_pred))