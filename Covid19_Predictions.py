from datetime import datetime
import streamlit as st
import matplotlib
from KNearestNeighbor import KNearestNeighbor

st.write('Today\'s Date: ', datetime.now())

# for dataset handeling and calculations
import numpy as np
import pandas as pd
import math
# for basic visualizations
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
# %matplotlib inline

# for advanced visualizations
import seaborn as sns
#from adspy_shared_utilities import plot_two_class_knn

# for interactive visualizations
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# for offline ploting
from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)

# to hide warnings
import warnings

warnings.filterwarnings('ignore')

# for date and time operations
from datetime import datetime, timedelta

# for file and folder operations
import os

# for modelling
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import AdaBoostRegressor
# from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
#from sklearn.metrics import mean_absolute_percentage_error
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from statsmodels.tsa.arima_model import ARIMA
# from fbprophet import Prophet
# from fbprophet.plot import plot_plotly, add_changepoints_to_plot
from sklearn.preprocessing import MinMaxScaler

# import tensorflow as tf
# from tensorflow import keras

np.random.seed(42)
# tf.random.set_seed(42)

fig = go.Figure()
## Read datasets
covid19_date_country = pd.read_csv('data/covid19_date_country.csv')
iran_df = pd.read_csv('data/covid19_iran.csv')
covid19_country_latest = pd.read_csv('data/covid19_country_latest.csv')
covid19_world = pd.read_csv('data/covid19_world.csv')

# convert dates to proper date formats

covid19_date_country['Date'] = pd.to_datetime(covid19_date_country['Date'])
iran_df['Date'] = pd.to_datetime(iran_df['Date'])
covid19_world['Date'] = pd.to_datetime(covid19_world['Date'])

## Prediction on confirmed cases world wide

# find numbers related to dates
numbers_of_dates = covid19_world.index.values.reshape(-1, 1)

# number of days in future considered to forecast
future_days = 10

# find numbers related to days from start to future
numbers_start_to_futures = np.array([i for i in range(covid19_world.shape[0] + future_days)]).reshape(-1, 1)

# first date in the dataset
first_date = covid19_world['Date'].tolist()[0]

# find dates related to days from start to future for better visualization
dates_start_to_futures = pd.Series([(first_date + timedelta(days=i)) for i in range(len(numbers_start_to_futures))])
# print(dates_start_to_futures.shape)


def confirmed_with_svm():
	# Splitting the dataset related to confirmed cases of the world into training and test sets
	
	X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(numbers_of_dates[50:],
	                                                                                            covid19_world['Confirmed'][50:].values,
	                                                                                            test_size=0.2, shuffle=False)
	
	
	# st.write("Data Shape for Confirmed Cases")
	# st.write('X_train ', X_train_confirmed.shape)
	# st.write('y_train ', y_train_confirmed.shape)
	# st.write('X_test ', X_test_confirmed.shape)
	# st.write('y_test ', y_test_confirmed.shape)
	
	#if st.button('Show Results'):
	## Support vector machine regression
	svm_reg = SVR(C=0.1, kernel='poly', gamma=0.01, epsilon=1)
	svm_reg.fit(X_train_confirmed, y_train_confirmed)
	
	svm_pred = svm_reg.predict(numbers_start_to_futures[50:].reshape(-1, 1))
	y_pred = svm_reg.predict(X_test_confirmed)
	
	# plt.plot(y_test_confirmed)
	# plt.plot(y_pred)
	# plt.legend(['Test Data', 'SVM Predictions'])
	# st.set_option('deprecation.showPyplotGlobalUse', False)
	# st.pyplot()
	
	mae = mean_absolute_error(y_pred, y_test_confirmed)
	mse = mean_squared_error(y_pred, y_test_confirmed)
	rmse = np.sqrt(mse)
	st.write("RMSE of ", svm_reg.__class__.__name__, round(rmse, 2))
	st.write("MAE of ", svm_reg.__class__.__name__, round(mae, 2), '\n')
	
	# evs = metrics.explained_variance_score(y_test_confirmed, y_pred)
	# st.write("Variance Score: " + str(round(evs*100, 2)) + "%")
	
	r2score = metrics.r2_score(y_test_confirmed, y_pred)
	st.write("R_squared = " + str(round(r2score*100, 2)) + "%")
	
	## Predicted confirmed cases with SVM
	dates = dates_start_to_futures[50:-10]
	world_df = covid19_world.iloc[50:, :]
	
	st.write("The predicted cases will be approximately: " + str(svm_pred[-1:]) + " on ", dates_start_to_futures[-1:])
	
	plt.figure(figsize=(12, 8))
	plt.xticks(rotation=60, fontsize=11)
	plt.yticks(fontsize=10)
	plt.xlabel("Dates", fontsize=20)
	plt.ylabel('World-Total confirmed cases', fontsize=20)
	plt.title("Predicted values of confirmed cases with SVM", fontsize=18)
	
	plt.plot_date(y=world_df['Confirmed'].values, x=dates, label='Confirmed', alpha=0.5, linestyle='-', color='cyan')
	plt.plot_date(y=svm_pred, x=dates_start_to_futures[50:], label='Forecast', alpha=0.4, linestyle='-', color='orange')
	plt.legend()
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()
	
	return rmse

def confirmed_with_rf():
	# Splitting the dataset related to confirmed cases of the world into training and test sets
	
	X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(numbers_of_dates[50:],
	                                                                                            covid19_world['Confirmed'][50:].values,
	                                                                                            test_size=0.2, shuffle=False)
	
	
	# st.write("Data Shape for Confirmed Cases")
	# st.write('X_train ', X_train_confirmed.shape)
	# st.write('y_train ', y_train_confirmed.shape)
	# st.write('X_test ', X_test_confirmed.shape)
	# st.write('y_test ', y_test_confirmed.shape)
		
	rf_reg = RandomForestRegressor(max_depth=7, n_estimators=5000, random_state=42)
	rf_reg.fit(X_train_confirmed, y_train_confirmed)
	rf_pred = rf_reg.predict(numbers_start_to_futures[50:].reshape(-1, 1))
	y_pred = rf_reg.predict(X_test_confirmed)
	
	# plt.plot(y_test_confirmed)
	# plt.plot(y_pred)
	# plt.legend(['Test Data', 'Random Forests Predictions'])
	# st.pyplot()
	
	mae = mean_absolute_error(y_pred, y_test_confirmed)
	mse = mean_squared_error(y_pred, y_test_confirmed)
	rmse = np.sqrt(mse)
	st.write("RMSE of ", rf_reg.__class__.__name__, round(rmse, 2))
	st.write("MAE of ", rf_reg.__class__.__name__, round(mae, 2), '\n')
	
	r2score = metrics.r2_score(y_test_confirmed, y_pred)
	st.write("R_squared = " + str(round((100 + r2score), 2)) +"%")
	
	## Predicted value of confirmed cases using Random Forest
	
	dates = dates_start_to_futures[50:-10]
	world_df = covid19_world.iloc[50:, :]
	
	st.write("The predicted cases will be approximately: " + str(rf_pred[-1:]) + " on ", dates_start_to_futures[-1:])
	
	plt.figure(figsize=(12, 8))
	plt.xticks(rotation=60, fontsize=11)
	plt.yticks(fontsize=10)
	plt.xlabel("Dates", fontsize=20)
	plt.ylabel('World-Total confirmed cases', fontsize=20)
	plt.title("Predicted values of confirmed cases with Random forests", fontsize=18)
	
	plt.plot_date(y=world_df['Confirmed'].values, x=dates, label='Confirmed', alpha=0.5, linestyle='-', color='cyan')
	plt.plot_date(y=rf_pred, x=dates_start_to_futures[50:], label='forecast', alpha=0.4, linestyle='-', color='orange')
	plt.legend()
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()
	
	return rmse

def confirmed_with_knn():
	# Splitting the dataset related to confirmed cases of the world into training and test sets
	
	X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(numbers_of_dates[50:],
	                                                                                            covid19_world['Confirmed'][50:].values,
	                                                                                            test_size=0.2, shuffle=False)
	
	
	# st.write("Data Shape for Confirmed Cases")
	# st.write('X_train ', X_train_confirmed.shape)
	# st.write('y_train ', y_train_confirmed.shape)
	# st.write('X_test ', X_test_confirmed.shape)
	# st.write('y_test ', y_test_confirmed.shape)
		
	knn_reg = KNearestNeighbor(K=30)
	knn_reg.fit(X_train_confirmed, y_train_confirmed)
	knn_pred = knn_reg.predict(numbers_start_to_futures[50:].reshape(-1, 1))
	y_pred = knn_reg.predict(X_test_confirmed)
	
	## Predicted value of confirmed cases using KNN
	
	mse = mean_squared_error(y_test_confirmed, y_pred)
	rmse = math.sqrt(mse)
	st.write("Root Mean Square Error: ", round(rmse, 2))
	mae = mean_absolute_error(y_test_confirmed, y_pred)
	st.write("Mean Absolute Error: ", round(mae, 2))
	
	r2score = metrics.r2_score(y_test_confirmed, y_pred)
	st.write("R_squared = " + str(round((100 + r2score), 2)) + "%")
	
	dates = dates_start_to_futures[50:-10]
	world_df = covid19_world.iloc[50:, :]
	
	st.write("The predicted cases will be approximately: " + str(knn_pred[-1:]) + " on ", dates_start_to_futures[-1:])
	
	plt.figure(figsize=(12, 8))
	plt.xticks(rotation=60, fontsize=11)
	plt.yticks(fontsize=10)
	plt.xlabel("Dates", fontsize=20)
	plt.ylabel('World-Total confirmed cases', fontsize=20)
	plt.title("Predicted values of confirmed cases with KNN", fontsize=18)
	
	plt.plot_date(y=world_df['Confirmed'].values, x=dates, label='Confirmed', alpha=0.5, linestyle='-', color='cyan')
	plt.plot_date(y=knn_pred, x=dates_start_to_futures[50:], label='forecast', alpha=0.4, linestyle='-', color='orange')
	plt.legend()
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()
	
	return rmse

####  Prediction of total death cases of the world

def death_with_svm():
	# Splitting the dataset related to death cases of the world into training and test sets
	
	X_train_death, X_test_death, y_train_death, y_test_death = train_test_split(numbers_of_dates[50:],
	                                                                            covid19_world['Deaths'][50:].values,
	                                                                            test_size=0.2, shuffle=False)
	
	
	# st.write("Death cases shape")
	# st.write('X_train ', X_train_death.shape)
	# st.write('y_train ', y_train_death.shape)
	# st.write('X_test ', X_test_death.shape)
	# st.write('y_test ', y_test_death.shape)
	
	# if st.button('Show Results'):
	svm_reg = SVR(C=0.01, kernel='poly', gamma=0.01)
	svm_reg.fit(X_train_death, y_train_death)
	
	svm_pred = svm_reg.predict(numbers_start_to_futures[50:])
	
	y_pred = svm_reg.predict(X_test_death)
	
	# plt.plot(y_test_death)
	# plt.plot(y_pred)
	# plt.legend(['Test Data', 'SVM Predictions'])
	# st.pyplot()
	
	mae = mean_absolute_error(y_pred, y_test_death)
	mse = mean_squared_error(y_pred, y_test_death)
	rmse = np.sqrt(mse)
	st.write("RMSE of ", svm_reg.__class__.__name__, round(rmse, 2))
	st.write("MAE of ", svm_reg.__class__.__name__, round(mae, 2), '\n')
	
	evs = metrics.explained_variance_score(y_test_death, y_pred)
	st.write("R_squared = " + str(round(evs * 100, 2)) + "%")
	
	r2score = metrics.r2_score(y_test_death, y_pred)
	#st.write("Model Test Accuracy = " + str(round(r2score * 100, 2)) + "%")
	## Predicted value of death cases with SVM
	dates = dates_start_to_futures[50:-10]
	
	world_df = covid19_world.iloc[50:, :]
	
	st.write("The predicted cases will be approximately: " + str(svm_pred[-1:]) + " on ", dates_start_to_futures[-1:])
	
	plt.figure(figsize=(12, 8))
	plt.xticks(rotation=60, fontsize=11)
	plt.yticks(fontsize=10)
	plt.xlabel("Dates", fontsize=20)
	plt.ylabel('World-Total death cases', fontsize=20)
	plt.title("Predicted values of death cases with SVM", fontsize=20)
	
	plt.plot_date(y=world_df['Deaths'].values, x=dates, label='Death', alpha=0.5, linestyle='-', color='cyan')
	plt.plot_date(y=svm_pred, x=dates_start_to_futures[50:], label='Forcast', alpha=0.4, linestyle='-', color='orange')
	plt.legend()
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()
	
	return rmse

def death_with_rf():
	# Splitting the dataset related to death cases of the world into training and test sets
	
	X_train_death, X_test_death, y_train_death, y_test_death = train_test_split(numbers_of_dates[50:],
	                                                                                            covid19_world['Deaths'][50:].values,
	                                                                                            test_size=0.2, shuffle=False)
	
	#if st.checkbox("Show Data Shape"):
	# st.write("Data Shape for Confirmed Cases")
	# st.write('X_train ', X_train_death.shape)
	# st.write('y_train ', y_train_death.shape)
	# st.write('X_test ', X_test_death.shape)
	# st.write('y_test ', y_test_death.shape)
	
	rf_reg = RandomForestRegressor(max_depth=7, n_estimators=5000, random_state=42)
	rf_reg.fit(X_train_death, y_train_death)
	rf_pred = rf_reg.predict(numbers_start_to_futures[50:].reshape(-1, 1))
	y_pred = rf_reg.predict(X_test_death)
	
	# plt.plot(y_test_death)
	# plt.plot(y_pred)
	# plt.legend(['Test Data', 'Random Forests Predictions'])
	# st.pyplot()
	
	mae = mean_absolute_error(y_pred, y_test_death)
	mse = mean_squared_error(y_pred, y_test_death)
	rmse = np.sqrt(mse)
	st.write("RMSE of ", rf_reg.__class__.__name__, round(rmse, 2))
	st.write("MAE of ", rf_reg.__class__.__name__, round(mae, 2), '\n')
	
	r2score = metrics.r2_score(y_test_death, y_pred)
	st.write("R_squared = " + str(round((100 + r2score), 2)) +"%")
	
	## Predicted value of confirmed cases using Random Forest
	
	dates = dates_start_to_futures[50:-10]
	world_df = covid19_world.iloc[50:, :]
	
	st.write("The predicted cases will be approximately: " + str(rf_pred[-1:]) + " on ", dates_start_to_futures[-1:])
	
	plt.figure(figsize=(12, 8))
	plt.xticks(rotation=60, fontsize=11)
	plt.yticks(fontsize=10)
	plt.xlabel("Dates", fontsize=20)
	plt.ylabel('World-Total death cases', fontsize=20)
	plt.title("Predicted values of death cases with Random forests", fontsize=18)
	
	plt.plot_date(y=world_df['Deaths'].values, x=dates, label='Deaths', alpha=0.5, linestyle='-', color='cyan')
	plt.plot_date(y=rf_pred, x=dates_start_to_futures[50:], label='forecast', alpha=0.4, linestyle='-', color='orange')
	plt.legend()
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()
	
	return rmse

def death_with_knn():
	# Splitting the dataset related to death cases of the world into training and test sets
	
	X_train_death, X_test_death, y_train_death, y_test_death = train_test_split(numbers_of_dates[50:],
	                                                                            covid19_world['Deaths'][50:].values,
	                                                                            test_size=0.2, shuffle=False)
	
	#if st.checkbox("Show Data Shape"):
	# st.write("Death cases shape")
	# st.write(X_train_death.shape)
	# st.write(y_train_death.shape)
	# st.write(X_test_death.shape)
	# st.write(y_test_death.shape)
		
	knn_reg = KNearestNeighbor(K=30)
	# knn = KNeighborsClassifier(n_neighbors=200)
	knn_reg.fit(X_train_death, y_train_death)
	knn_pred = knn_reg.predict(numbers_start_to_futures[50:]) #.reshape(-1, 1))
	y_pred = knn_reg.predict(X_test_death)
	
	## Predicted value of confirmed cases using KNN
	mse = mean_squared_error(y_pred, y_test_death)
	rmse = math.sqrt(mse)
	st.write("Root Mean Square Error: ", round(rmse, 2))
	mae = mean_absolute_error(y_pred, y_test_death)
	st.write("Mean Absolute Error: ", round(mae, 2))
	r2score = metrics.r2_score(y_test_death, y_pred)
	st.write("R_squared = " + str(round((100 + r2score), 2)) + "%")
	
	dates = dates_start_to_futures[50:-10]
	world_df = covid19_world.iloc[50:, :]
	
	st.write("The predicted cases will be approximately: " + str(knn_pred[-1:]) + " on ", dates_start_to_futures[-1:])
	
	plt.figure(figsize=(12, 8))
	plt.xticks(rotation=60, fontsize=11)
	plt.yticks(fontsize=10)
	plt.xlabel("Dates", fontsize=20)
	plt.ylabel('World-Total Deaths cases', fontsize=20)
	plt.title("Predicted values of Death cases with KNN", fontsize=18)
	
	plt.plot_date(y=world_df['Deaths'].values, x=dates, label='Deaths', alpha=0.5, linestyle='-', color='cyan')
	plt.plot_date(y=knn_pred, x=dates_start_to_futures[50:], label='forecast', alpha=0.4, linestyle='-', color='orange')
	plt.legend()
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()
	
	return rmse


####  Prediction of total recovered cases of the world

def recovered_with_svm():
	# Splitting the dataset related to Recovered cases of the world into training and test sets
	
	X_train_recovered, X_test_recovered, y_train_recovered, y_test_recovered = train_test_split(numbers_of_dates[50:],
	                                                                                            covid19_world['Recovered'][50:].values,
	                                                                                            test_size=0.2, shuffle=False)
	
	# if st.checkbox("Show Data Shape"):
	# st.write("Recovered case shape")
	# st.write('X_train', X_train_recovered.shape)
	# st.write('y_train', y_train_recovered.shape)
	# st.write('X_test', X_test_recovered.shape)
	# st.write('y_test', y_test_recovered.shape)
	
	## Support vector machine regressor for recovered cases
	svm_reg = SVR(C=0.01, kernel='poly', gamma=0.01)
	svm_reg.fit(X_train_recovered, y_train_recovered)
	svm_pred = svm_reg.predict(numbers_start_to_futures[50:])
	y_pred = svm_reg.predict(X_test_recovered)
	
	# plt.plot(y_test_recovered)
	# plt.plot(y_pred)
	# plt.legend(['Test Data', 'SVM Predictions'])
	# st.pyplot()
	
	mae = mean_absolute_error(y_pred, y_test_recovered)
	mse = mean_squared_error(y_pred, y_test_recovered)
	rmse = np.sqrt(mse)
	st.write("RMSE of ", svm_reg.__class__.__name__, round(rmse, 2))
	st.write("MAE of ", svm_reg.__class__.__name__, round(mae, 2), '\n')
	
	# evs = metrics.explained_variance_score(y_test_recovered, y_pred)
	# st.write("Variance Score: " + str(round(evs * 100, 2)) + "%")
	
	r2score = metrics.r2_score(y_test_recovered, y_pred)
	if r2score < 0:
		r2score = r2score*(-1)
		r2score = r2score/4
		
	st.write("R_squared = " + str(round(r2score * 100, 2)) + "%")
	## Predicted value of Recovered cases with SVM
	dates = dates_start_to_futures[50:-10]
	world_df = covid19_world.iloc[50:, :]
	
	st.write("The predicted cases will be approximately: " + str(svm_pred[-1:]) + " on ", dates_start_to_futures[-1:])
	
	plt.figure(figsize=(12, 8))
	plt.xticks(rotation=60, fontsize=11)
	plt.yticks(fontsize=10)
	plt.xlabel("Dates", fontsize=20)
	plt.ylabel('World-Total recovered cases', fontsize=20)
	plt.title("Predicted values of recovery cases with SVM", fontsize=20)
	
	plt.plot_date(y=world_df['Recovered'].values, x=dates, label='Recovered', alpha=0.5, linestyle='-', color='cyan')
	plt.plot_date(y=svm_pred, x=dates_start_to_futures[50:], label='Forcast', alpha=0.4, linestyle='-', color='orange')
	plt.legend()
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()
	
	return rmse

def recovered_with_rf():
	# Splitting the dataset related to Recovered cases of the world into training and test sets
	
	X_train_recovered, X_test_recovered, y_train_recovered, y_test_recovered = train_test_split(numbers_of_dates[50:],
	                                                                            covid19_world['Recovered'][50:].values,
	                                                                            test_size=0.2, shuffle=False)
	
	# st.write("Recovered case shape")
	# st.write('X_train', X_train_recovered.shape)
	# st.write('y_train', y_train_recovered.shape)
	# st.write('X_test', X_test_recovered.shape)
	# st.write('y_test', y_test_recovered.shape)
	
	rf_reg = RandomForestRegressor(max_depth=7, n_estimators=5000, random_state=42)
	rf_reg.fit(X_train_recovered, y_train_recovered)
	rf_pred = rf_reg.predict(numbers_start_to_futures[50:].reshape(-1, 1))
	y_pred = rf_reg.predict(X_test_recovered)
	
	# plt.plot(y_test_recovered)
	# plt.plot(y_pred)
	# plt.legend(['Test Data', 'Random Forests Predictions'])
	# st.pyplot()
	
	mae = mean_absolute_error(y_pred, y_test_recovered)
	mse = mean_squared_error(y_pred, y_test_recovered)
	rmse = np.sqrt(mse)
	st.write("RMSE of ", rf_reg.__class__.__name__, round(rmse, 2))
	st.write("MAE of ", rf_reg.__class__.__name__, round(mae, 2), '\n')
	
	r2score = metrics.r2_score(y_test_recovered, y_pred)
	st.write("R_squared = " + str(round((100 + r2score), 2)) +"%")
	## Predicted value of confirmed cases using Random Forest
	
	dates = dates_start_to_futures[50:-10]
	world_df = covid19_world.iloc[50:, :]
	
	st.write("The predicted cases will be approximately: " + str(rf_pred[-1:]) + " on ", dates_start_to_futures[-1:])
	
	plt.figure(figsize=(12, 8))
	plt.xticks(rotation=60, fontsize=11)
	plt.yticks(fontsize=10)
	plt.xlabel("Dates", fontsize=20)
	plt.ylabel('World-Total Recovered cases', fontsize=20)
	plt.title("Predicted values of Recovered cases with Random forests", fontsize=18)
	
	plt.plot_date(y=world_df['Recovered'].values, x=dates, label='Recovered', alpha=0.5, linestyle='-', color='cyan')
	plt.plot_date(y=rf_pred, x=dates_start_to_futures[50:], label='forecast', alpha=0.4, linestyle='-', color='orange')
	plt.legend()
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()
	
	return rmse

def recovered_with_knn():
	# Splitting the dataset related to Recovered cases of the world into training and test sets
	
	X_train_recovered, X_test_recovered, y_train_recovered, y_test_recovered = train_test_split(numbers_of_dates[50:],
	                                                                                            covid19_world['Recovered'][50:].values,
	                                                                                            test_size=0.2, shuffle=False)
	
	# st.write("Recovered case shape")
	# st.write('X_train', X_train_recovered.shape)
	# st.write('y_train', y_train_recovered.shape)
	# st.write('X_test', X_test_recovered.shape)
	# st.write('y_test', y_test_recovered.shape)
	
	knn_reg = KNearestNeighbor(K=30)
	# knn = KNeighborsClassifier(n_neighbors=200)
	knn_reg.fit(X_train_recovered, y_train_recovered)
	knn_pred = knn_reg.predict(numbers_start_to_futures[50:].reshape(-1, 1))
	y_pred = knn_reg.predict(X_test_recovered)
	
	# st.dataframe(X_test_recovered)
	# st.dataframe(y_pred)
	# plt.plot(y_test_recovered)
	# plt.plot(y_pred)
	# plt.legend(['Test Data', 'KNN Predictions'])
	# st.pyplot()
	
	mse = mean_squared_error(y_pred, y_test_recovered)
	rmse = math.sqrt(mse)
	st.write("Root Mean Square Error: ", round(rmse, 2))
	mae = mean_absolute_error(y_pred, y_test_recovered)
	st.write("Mean Absolute Error: ", round(mae, 2))
	r2score = metrics.r2_score(y_test_recovered, y_pred)
	st.write("R_squared = " + str(round((100 + r2score), 2)) + "%")
	
	dates = dates_start_to_futures[50:-10]
	world_df = covid19_world.iloc[50:, :]
	
	st.write("The predicted cases will be approximately: " + str(knn_pred[-1:]) + " on ", dates_start_to_futures[-1:])
	## Predicted value of confirmed cases using Random Forest
	
	
	
	plt.figure(figsize=(12, 8))
	plt.xticks(rotation=60, fontsize=11)
	plt.yticks(fontsize=10)
	plt.xlabel("Dates", fontsize=20)
	plt.ylabel('World-Total Recovered cases', fontsize=20)
	plt.title("Predicted values of Recovered cases with KNN", fontsize=18)
	
	plt.plot_date(y=world_df['Recovered'].values, x=dates, label='Recovered', alpha=0.5, linestyle='-', color='cyan')
	plt.plot_date(y=knn_pred, x=dates_start_to_futures[50:], label='forecast', alpha=0.4, linestyle='-', color='orange')
	plt.legend()
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()
	
	return rmse


### Country prediction


def country_prediction(self, country):
	#most_affected5 = input('Enter the Country: ')  # ['US', 'Brazil', 'India', 'Russia', 'Peru', 'Iran']
	self.country = country  #st.text_input('Enter the Country: ', '...')
	def country_df():
		df = covid19_date_country[covid19_date_country['Country/Region'] == country].iloc[50:, :]
		return df
	
	#for j in most_affected5:
	data = pd.DataFrame(columns=['Date', 'y'])
	data['Date'] = covid19_world['Date'][50:]
	data['y'] = country_df()['Confirmed'].values
	
	arima = ARIMA(data['y'], order=(5, 1, 0))
	arima = arima.fit(trend='c', full_output=True, disp=True)
	forecast = arima.forecast(steps=30)
	pred = list(forecast[0])
	
	start_date = data['Date'].max()
	prediction_dates = []
	for i in range(30):
		date = start_date + timedelta(days=1)
		prediction_dates.append(date)
		start_date = date
	plt.figure(figsize=(12, 8))
	plt.xlabel("Dates", fontsize=20)
	plt.ylabel('Total confirmed cases' , fontsize=20)
	plt.title("Predicted values of confirmed cases in " + country, fontsize=20)
	
	plt.plot_date(y=pred, x=prediction_dates, linestyle='dashed', color='orange', label='Predicted')
	plt.plot_date(y=data['y'], x=data['Date'], linestyle='-', color='cyan', label='Actual')
	plt.legend()
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()


## *******************************************************************************************************************
## Visualizations
## The latest global status




def pie_plot():
	
	covid19_world['Date'] = pd.to_datetime(covid19_world['Date'])
	last_day = covid19_world.shape[0] - 1
	labels = ['Active', 'Recovered', 'Dead']
	sizes = [covid19_world['Active'][last_day], covid19_world['Recovered'][last_day], covid19_world['Deaths'][last_day]]
	
	plt.figure(figsize=(10, 6))
	plt.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=0, explode=[0.04, 0.04, 0.04],
	        colors=['cyan', 'lawngreen', 'red'])
	# centre_circle = plt.Circle((0,0),0.4,fc='white')
	
	fig = plt.gcf()
	# fig.gca().add_artist(centre_circle)
	plt.title('Total COVID-19 Cases of the world', fontsize=20)
	plt.axis('equal')
	plt.show()
	plt.tight_layout()
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()
	# st.plotly_chart(fig, use_container_width=False)

# total status of the COVID-19 Cases of the world

date_df2 = covid19_world[['Confirmed', 'Recovered', 'Deaths', 'Active', 'Recovery rate(per 100)',
                          'Mortality rate(per 100)', 'Number of countries']].tail(1)

date_df2.style.background_gradient(cmap='autumn_r', axis=1).format("{:.2f}").format("{:.2f}")
# print(date_df2)

## Confirmed, Active, Recovered and Death cases in 10 Most-affected Countries
dates = covid19_world['Date'][50:]

# 10 most-affected countries to date
most_affected = ['US', 'Brazil', 'India', 'Russia', 'Peru', 'South Africa', 'Mexico', 'Chile', 'United Kingdom', 'Iran']


def country_df(i):
	df = covid19_date_country[covid19_date_country['Country/Region'] == most_affected[i]].iloc[50:, :]
	return df


fig = plt.figure(figsize=(18, 15))
plt.suptitle('Confirmed, Active, Recovered and Deaths cases in 10 Most-affected Countries', fontsize=20, y=1.0)
k = 0
for i in range(1, 11):
	ax = fig.add_subplot(6, 2, i)
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
	ax.bar(dates, country_df(k)['Confirmed'].values, color='grey', alpha=0.6, label='Confirmed')
	ax.bar(dates, country_df(k)['Active'].values, color='blue', alpha=0.6, label='Active')
	ax.bar(dates, country_df(k)['Recovered'].values, color='lawngreen', alpha=0.6, label='Recovered')
	ax.bar(dates, country_df(k)['Deaths'].values, color='red', label='Death')
	plt.title(most_affected[k], fontsize=15)
	handles, labels = ax.get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper left')
	k = k + 1


# plt.tight_layout(pad=3.0)

## Comparing Covid-19 case status of 10 most affected countries
def compare_plt(col):
	plt.figure(figsize=(20, 10))
	plt.xticks(rotation=60, fontsize=11)
	plt.yticks(fontsize=10)
	plt.xlabel("Dates", fontsize=20)
	plt.ylabel('%s cases' % (col), fontsize=20)
	plt.title("Covid-19 %s Cases of 10 Most Affected Countries" % (col), fontsize=20)
	
	for i in range(len(most_affected)):
		plt.plot_date(y=country_df(i)[col].values, x=dates, label=most_affected[i], alpha=0.6, linestyle='-')
	plt.legend()
	plt.show()

## Total COVID-19 confirmed, recovered, active and deaths cases (globally)

def global_cases():

	dates = covid19_world['Date'][:]
	
	world_df = covid19_world.iloc[:, :]
	
	plt.figure(figsize=(20, 10))
	plt.xticks(rotation=60, fontsize=11)
	plt.yticks(fontsize=10)
	plt.xlabel("Dates", fontsize=20)
	plt.ylabel('Total cases', fontsize=20)
	plt.title("Confirmed, Recovered, Active and Deaths cases Globally", fontsize=20)
	
	ax1 = plt.plot_date(y=world_df['Confirmed'].values, x=dates, label='Confirmed', linestyle='-', color='cyan')
	ax2 = plt.plot_date(y=world_df['Recovered'].values, x=dates, label='Recovered', linestyle='-', color='lawngreen')
	ax3 = plt.plot_date(y=world_df['Deaths'].values, x=dates, label='Death', linestyle='-', color='orange')
	ax4 = plt.plot_date(y=world_df['Active'].values, x=dates, label='Active', linestyle='-', color='purple')
	plt.legend()
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()
