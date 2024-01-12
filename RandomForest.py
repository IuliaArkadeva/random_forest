######################################################################################
# Random Forest implementation using python
# Instructor: Prof. Loubna Ali
# Berlin School of Business and Innovation
# Data Analytics Group 2 - Oct '23
######################################################################################


#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#read csv into features
features = pd.read_csv('C:/Users/aliba/Desktop/cs/randforest_python_weather/temps.csv')
#print(features.head(5))
#print(features.describe())

#adding binary day fields (might not be necessary)
features = pd.get_dummies(features)
#print(features.head(5))

#targets a.k.a. labels
labels = np.array(features['actual'])

#dropping targets from features
features = features.drop('actual', axis = 1)

#Saving field names for later 
feature_list = list(features.columns)

#Transforming into numpy array
features = np.array(features)

#Training splits
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
#print('Training features shape:', train_features.shape)
#print('Training labels shape:', train_labels.shape)
#print('Testing features shape:', test_features.shape)
#print('Testing labels shape:', test_labels.shape)

#Base predictions & error
baseline_preds = test_features[:, feature_list.index('average')]
baseline_errors = abs(baseline_preds - test_labels)
#print('Average baseline error: ', round(np.mean(baseline_errors), 2))

#Random forest begins
rf = RandomForestRegressor(n_estimators= 1000, random_state= 42)
rf.fit(train_features, train_labels)

#Random forest predictions & error
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
#print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

#Mean absolute percentage error(MAPE) & Accuracy
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
#print('Accuracy:', round(accuracy, 2), '%.')

