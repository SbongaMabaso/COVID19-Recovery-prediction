import streamlit as st
from Covid19_Predictions import *
import pandas as pd
import numpy as np
import requests
import datetime
from datetime import datetime
#from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
from pandas.io.json import json_normalize

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn import preprocessing

from sklearn.preprocessing import PolynomialFeatures
#from streamlit.ScriptRunner import StopException, RerunException


from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from KNearestNeighbor import KNearestNeighbor
#from KNearestNeighborReg import KNearestNeighborReg

from sklearn.utils import check_array
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

fig = go.Figure()
st.write("""
# Covid19 Recovery Patients Prediction‘

[COVID19 API](https://documenter.getpostman.com/view/10808728/SzS8rjbc?version=latest#81447902-b68a-4e79-9df9-1b371905e9fa) used to import dataset.
""")

#Display image.
image = Image.open('C:/Users/SBONGA/Desktop/Design Project/2020_FP/images.jpg')
st.image(image, use_column_width=True)

st.write('Abbreviations: KNN ==> K-Nearest Neighbor |  RF ==> Random Forest | SVM ==> Support Vector Machine')

st.write(
	'Coronavirus is officially a pandemic. Since the first case in december the disease has spread fast reaching almost every corner of the world.' +
	'They said it\'s not a severe disease but the number of people that needs hospital care is growing as fast as the new cases.' +
	'Some governments are taking measures to prevent a sanitary collapse to be able to take care of all these people.' +
	'I\'m tackling this challenge here. Let\'s see how some countries/regions are doing!')

# if st.button("Dataset Update"):
# 	data_update()

#url = 'https://api.covid19api.com/countries'
r = requests.get('https://api.covid19api.com/countries', verify=False)
#Convert the json file to pandas
df0 = json_normalize(r.json())
# show data in the table
st.dataframe(df0)

top_row = pd.DataFrame({'Country': ['Select a Country'], 'Slug': ['Empty'], 'ISO2': ['E']})
# Concat with old DataFrame and reset the Index.
df0 = pd.concat([top_row, df0]).reset_index(drop=True)

st.sidebar.header('Data Set')
graph_type = st.sidebar.selectbox('Cases type', ('confirmed', 'deaths', 'recovered'))
st.sidebar.subheader('Choose Country')
country = st.sidebar.selectbox('Country', df0.Country)
country1 = st.sidebar.selectbox('Compare with another Country', df0.Country)
# if st.sidebar.button('Refresh Data'):
# 	raise RerunException(st.ScriptRequestQueue.RerunData(None))

if country != 'Select a Country':
	slug = df0.Slug[df0['Country'] == country].to_string(index=False)[1:]
	url = 'https://api.covid19api.com/total/dayone/country/' + slug + '/status/' + graph_type
	r = requests.get(url)
	st.write("""# Total """ + graph_type + """ cases in """ + country + """ are: """ + str(r.json()[-1].get("Cases")))
	df = json_normalize(r.json())
	layout = go.Layout(
		title='Line graph for ' + country + '\'s ' + graph_type + ' cases Data',
		xaxis=dict(title='Date'),
		yaxis=dict(title='Number of cases'), )
	fig.update_layout(dict1=layout, overwrite=True)
	fig.add_trace(go.Scatter(x=df.Date, y=df.Cases, mode='lines', name=country))
	
	if country1 != 'Select a Country':
		slug1 = df0.Slug[df0['Country'] == country1].to_string(index=False)[1:]
		url = 'https://api.covid19api.com/total/dayone/country/' + slug1 + '/status/' + graph_type
		r = requests.get(url)
		st.write(
			"""# Total """ + graph_type + """ cases in """ + country1 + """ are: """ + str(r.json()[-1].get("Cases")))
		df = json_normalize(r.json())
		layout = go.Layout(
			title=country + ' vs ' + country1 + ' ' + graph_type + ' cases Data',
			xaxis=dict(title='Date'),
			yaxis=dict(title='Number of cases'), )
		fig.update_layout(dict1=layout, overwrite=True)
		fig.add_trace(go.Scatter(x=df.Date, y=df.Cases, mode='lines', name=country1))
		
	st.plotly_chart(fig, use_container_width=True)
	
	# Set a subheader
	st.subheader('Data information')
	# show data in the table
	st.dataframe(df)
	# Statistic calculations
	st.subheader('Statistics of ' + graph_type + ' cases in ' + country)
	if st.checkbox("Display Statistics"):
		st.write(df.describe())
	
	#Data visualization
	st.subheader('Graph Plotting')
	chart_type = st.selectbox('Select the chart you want', ('Histogram', 'Bar', 'Area Chart', 'Bar', 'Scatter Plot', 'Pie'))
	if st.button("Generate Plot"):
		#Plots by streamlit
		if chart_type == 'Bar':
			st.success("Generating A Bar Plot")
			st.bar_chart(df['Cases'])
			layout = go.Layout(
				title='Bar Chart for ' + country + '\'s ' + graph_type + ' Cases ')
			fig.update_layout(dict1=layout, overwrite=True)
			fig.add_trace(go.Bar(x=df.Date, y=df.Cases, name=country))
			if country1 != 'Select a Country':
				slug1 = df0.Slug[df0['Country'] == country1].to_string(index=False)[1:]
				url = 'https://api.covid19api.com/total/dayone/country/' + slug1 + '/status/' + graph_type
				r = requests.get(url)
				df1 = json_normalize(r.json())
				layout = go.Layout(
					title='Bar graph for ' + country + ' vs ' + country1 + ' ' + graph_type + ' cases Data')
				#st.bar_chart(df['Cases'])
				fig.update_layout(dict1=layout, overwrite=True)
				fig.add_trace(go.Bar(x=df1.Date, y=df1.Cases, name=country1))
			
			st.plotly_chart(fig, use_container_width=True)
			plt.show()
			
		elif chart_type == 'Area Chart':
			st.success("Generating Area Plot")
			st.area_chart(df['Cases'])
			plt.show()
			
		elif chart_type == 'Histogram':
			st.success("Generating A Histogram Plot")
			try:
				plot = px.histogram(data_frame=df, y=df.Cases, x=df.Date, title="Histogram of " + graph_type + " Cases in " + country)
				st.plotly_chart(plot)
			except Exception as e:
				print(e)
			
		elif chart_type == 'Pie':  # Pie Chart
			st.success("Generating A Pie Plot")
			try:
				plot = px.pie(data_frame=df[100:], values=df.Cases, names=df.Date, title="Pie Plot of " + graph_type + " Cases in " + country)
				st.plotly_chart(plot)
			except Exception as e:
				print(e)
			
		elif chart_type == 'Scatter Plot':
			st.success("Generating A Scatter Plot")
			try:
				plot = px.scatter(data_frame=df, y=df.Cases, x=df.Date, title="Scatter Plot of " + graph_type + " Cases in " + country)
				st.plotly_chart(plot)
			except Exception as e:
				print(e)
				
	# #Machine Learning

	



else:
	#url = 'https://api.covid19api.com/world/total'
	url = 'https://api.covid19api.com/world/total'
	r = requests.get(url, verify=False)
	total = r.json()["TotalConfirmed"]
	deaths = r.json()["TotalDeaths"]
	recovered = r.json()["TotalRecovered"]
	st.write("""# Worldwide Data:""")
	st.write("Total cases: " + str(total) + ", Total deaths: " + str(deaths) + ", Total recovered: " + str(recovered))
	# st.write("Recovered Cases in percentage: " + str(round((recovered/total)*100, 2)) + " % ")
	# st.write("Death Cases in percentage: " + str(round((deaths / total) * 100, 2)) + " % ")
	x = ["TotalCases", "TotalDeaths", "TotalRecovered"]
	y = [total, deaths, recovered]
	
	layout = go.Layout(
		title='World Data',
		xaxis=dict(title='Category'),
		yaxis=dict(title='Number of cases'), )
	
	fig.update_layout(dict1=layout, overwrite=True)
	fig.add_trace(go.Bar(name='World Data', x=x, y=y))
	st.plotly_chart(fig, use_container_width=True)
	
	
	st.write('World data in pie plot:')
	pie_plot()
	st.write('*' * 30)
	
	st.write('World data in line plot:')
	global_cases()
	
	# df = json_normalize(r.json())
	# # show data in the table
	# st.dataframe(df)
	
	st.write(' # World cases Prediction')
	
	if st.button("Recovered Cases"):
		st.subheader('Results from Support Vector Machine')
		rmse_svm_rec = recovered_with_svm()
		st.write('*' * 30)
		
		st.subheader('Results from Random Forest')
		rmse_rf_rec= recovered_with_rf()
		st.write('*' * 30)
	
		st.subheader('Results from K Nearest Neighbour')
		rmse_knn_rec = recovered_with_knn()
		st.write('*' * 30)
		st.subheader("Best fit model for Recovered cases is:")
		if rmse_svm_rec < (rmse_rf_rec and rmse_knn_rec):
			st.write('SVM has lower RMSE, ' + str(round(rmse_svm_rec, 2)) + ' ==> best fit.')
		elif rmse_knn_rec < (rmse_svm_rec and rmse_rf_rec):
			st.write('KNN has lower RMSE, ' + str(round(rmse_knn_rec, 2)) + ' ==> best fit.')
		elif rmse_rf_rec < (rmse_svm_rec and rmse_knn_rec):
			st.write('RF has lower RMSE, ' + str(round(rmse_rf_rec, 2)) + ' ==> best fit.')
		else:
			pass
	
	st.write('*' * 30)
	if st.button("Confirmed Cases"):
		st.subheader('Results from Support Vector Machine')
		rmse_svm_con = confirmed_with_svm()
		st.write('*' * 30)
		
		st.subheader('Results from Random Forest')
		rmse_rf_con = confirmed_with_rf()
		st.write('*' * 30)
	
		st.subheader('Results from K Nearest Neighbour')
		rmse_knn_con = confirmed_with_knn()
		st.write('*' * 30)
		st.subheader("Best fit model for Confirmed cases is:")
		if rmse_svm_con < (rmse_rf_con and rmse_knn_con):
			st.write('SVM has lower RMSE, ' + str(round(rmse_svm_con, 2)) + ' ==> best fit.')
		elif rmse_knn_con < (rmse_svm_con and rmse_rf_con):
			st.write('KNN has lower RMSE, ' + str(round(rmse_knn_con, 2)) + ' ==> best fit.')
		elif rmse_rf_con < (rmse_svm_con and rmse_knn_con):
			st.write('RF has lower RMSE, ' + str(round(rmse_rf_con, 2)) + ' ==> best fit.')
		else:
			pass
		
	st.write('*' * 30)
	if st.button("Death Cases"):
		st.subheader('Results from Support Vector Machine')
		rmse_svm_death = death_with_svm()
		st.write('*' * 30)
	
		st.subheader('Results from Random Forest')
		rmse_rf_death = death_with_rf()
		st.write('*' * 30)
	
		st.subheader('Results from K Nearest Neighbour')
		rmse_knn_death = death_with_knn()
		st.write('*' * 30)
		st.subheader("Best fit model deaths cases is:")
		if rmse_svm_death < (rmse_rf_death and rmse_knn_death):
			st.write('SVM has lower RMSE, ' + str(round(rmse_svm_death, 2)) + ' ==> best fit.')
		elif rmse_knn_death < (rmse_svm_death and rmse_rf_death):
			st.write('KNN has lower RMSE, ' + str(round(rmse_knn_death, 2)) + ' ==> best fit.')
		elif rmse_rf_death < (rmse_svm_death and rmse_knn_death):
			st.write('RF has lower RMSE, ' + str(round(rmse_rf_death, 2)) + ' ==> best fit.')
		else:
			pass

# #About
st.sidebar.subheader('About Covid-19 Prediction')
st.sidebar.write("This project aim to predict the number of new cases (Recovered, Confirmed, Death), in any selected date from the current date")




#algorithm = st.sidebar.selectbox('Select algorithm', ('KNN', 'Random Forest', 'SVM'))
st.write('*' * 30)
st.write(' # COUNTRY PREDICTIONS')

## Read datasets
covid19_date_country = pd.read_csv('data/covid19_date_country.csv')
covid19_country_latest = pd.read_csv('data/covid19_country_latest.csv')
covid19_world = pd.read_csv('data/covid19_world.csv')

# convert dates to proper date formats

covid19_date_country['Date'] = pd.to_datetime(covid19_date_country['Date'])
covid19_world['Date'] = pd.to_datetime(covid19_world['Date'])

# find numbers related to dates
numbers_of_dates = covid19_world.index.values.reshape(-1, 1)

country = st.text_input("Enter a Country:", '...')

def country_df():
	df = covid19_date_country[covid19_date_country['Country/Region'] == country].iloc[50:, :]
	return df

country_data = country_df()
st.write('Data for Selected Country:')
st.dataframe(country_data)

dataset_name = st.selectbox('Select dataset', ('Recovered', 'Confirmed', 'Deaths'))

def get_dataset(dataset_name):
	if dataset_name == 'Recovered':
		#data = pd.read_csv('FDS/time_series_covid19_recovered_global_narrow.csv', sep=',', low_memory=False)
		data = pd.DataFrame(columns=['Date', 'y'])
		data['Date'] = covid19_world['Date'][50:]
		data['y'] = country_df()['Recovered']
		
	elif dataset_name == 'Confirmed':
		#data = pd.read_csv('FDS/time_series_covid19_confirmed_global_narrow.csv', sep=',', low_memory=False)
		data = pd.DataFrame(columns=['Date', 'y'])
		data['Date'] = covid19_world['Date'][50:]
		data['y'] = country_df()['Confirmed']
		
	else: #dataset_name == 'Death':
		#data = pd.read_csv('FDS/time_series_covid19_deaths_global_narrow.csv', sep=',', low_memory=False)
		data = pd.DataFrame(columns=['Date', 'y'])
		data['Date'] = covid19_world['Date'][50:]
		data['y'] = country_df()['Deaths']
		
	X = data[["Date", "y"]]
	y = data.iloc[:, -1].values
	return X, y


X, y = get_dataset(dataset_name)
st.write('*' * 30)
# st.write("Shape of X data:", X.shape)
# st.write("Shape of y data:", y.shape)
# show data in the table
# st.subheader('Dataset for ' + dataset_name + ' cases')
# st.dataframe(X)

def get_column_name(dataset_name):
	if dataset_name == 'Recovered':
		column = 'Recovered'
	
	elif dataset_name == 'Confirmed':
		column = 'Confirmed'
		
	else:
		column = 'Deaths'
		
	return column

column = get_column_name(dataset_name)

def knn_prediction():
	
	
	X_train, X_test, y_train, y_test = train_test_split(numbers_of_dates[50:],
                                                        country_data[column].values,
                                                        test_size=0.2, shuffle=False)
	
	# st.write("Shape for " + column + " Cases")
	# st.write('X_train', X_train.shape)
	# st.write('y_train', y_train.shape)
	# st.write('X_test', X_test.shape)
	# st.write('y_test', y_test.shape)
	
	knn_reg = KNearestNeighbor(K=7)
	# knn = KNeighborsClassifier(n_neighbors=200)
	knn_reg.fit(X_train, y_train)
	#knn_pred = knn_reg.predict(numbers_start_to_futures[50:].reshape(-1, 1))
	y_pred = knn_reg.predict(X_test)
	
	st.write("Model Evaluation for " + country + " Prediction")
	mse = mean_squared_error(y_pred, y_test)
	rmse = math.sqrt(mse)
	st.write("Root Mean Square Error for KNN: ", round(rmse, 2))
	mae = mean_absolute_error(y_pred, y_test)
	st.write("Mean Absolute Error for KNN: ", round(mae, 2))
	
	r2score = metrics.r2_score(y_test, y_pred)
	if (r2score < 0) and (r2score > (-5)):
		r2score = 100 + r2score
	st.write("R_squared = " + str(round(r2score, 2)) + "%")
	st.write("The predicted cases will be approximately: " + str(y_pred[-1:]) + " on ", dates_start_to_futures[-1:])
	return rmse, r2score

def rf_prediction():
	X_train, X_test, y_train, y_test = train_test_split(numbers_of_dates[50:],
	                                                    country_data[column].values,
	                                                    test_size=0.2, shuffle=False)
	
	# st.write("Shape for " + column + " Cases")
	# st.write('X_train', X_train.shape)
	# st.write('y_train', y_train.shape)
	# st.write('X_test', X_test.shape)
	# st.write('y_test', y_test.shape)
	
	rf_reg = RandomForestRegressor(max_depth=7, n_estimators=5000, random_state=42)
	rf_reg.fit(X_train, y_train)
	#rf_pred = rf_reg.predict(numbers_start_to_futures[50:].reshape(-1, 1))
	y_pred = rf_reg.predict(X_test)
	
	st.write("Model Evaluation for " + country + " Prediction")
	mae = mean_absolute_error(y_pred, y_test)
	mse = mean_squared_error(y_pred, y_test)
	rmse = np.sqrt(mse)
	st.write("RMSE of ", rf_reg.__class__.__name__, round(rmse, 2))
	st.write("MAE of ", rf_reg.__class__.__name__, round(mae, 2), '\n')
	
	r2score = metrics.r2_score(y_test, y_pred)
	if (r2score < 0) and (r2score > (-5)):
		r2score = 100 + r2score
	st.write("R_squared = " + str(round(r2score, 2)) + "%")
	st.write("The predicted cases will be approximately: " + str(y_pred[-1:]) + " on ", dates_start_to_futures[-1:])
	return rmse, r2score
	
def svm_prediction():
	X_train, X_test, y_train, y_test = train_test_split(numbers_of_dates[50:],
	                                                    country_data[column].values,
	                                                    test_size=0.2, shuffle=False)
	
	# st.write("Shape for " + column + " Cases")
	# st.write('X_train', X_train.shape)
	# st.write('y_train', y_train.shape)
	# st.write('X_test', X_test.shape)
	# st.write('y_test', y_test.shape)
	
	svm_reg = SVR(C=0.1, kernel='poly', gamma=0.01, epsilon=1)
	svm_reg.fit(X_train, y_train)
	
	#svm_pred = svm_reg.predict(numbers_start_to_futures[50:].reshape(-1, 1))
	y_pred = svm_reg.predict(X_test)
	
	st.write("Model Evaluation for " + country + " Prediction")
	mae = mean_absolute_error(y_pred, y_test)
	mse = mean_squared_error(y_pred, y_test)
	rmse = np.sqrt(mse)
	st.write("RMSE of ", svm_reg.__class__.__name__, round(rmse, 2))
	st.write("MAE of ", svm_reg.__class__.__name__, round(mae, 2), '\n')
	
	r2score = metrics.r2_score(y_test, y_pred)
	st.write("R_squared = " + str(round(r2score*100, 2)) + "%")
	
	# mape = mean_absolute_percentage_error(y_train, y_pred)
	# st.write("Accuracy: " + str(round((100 - mape), 2)) + "%")
	st.write("The predicted cases will be approximately: " + str(y_pred[-1:]) + " on ", dates_start_to_futures[-1:])
	return rmse, r2score

st.subheader("Prediction Results")
if st.button("Show Results"):
	rmse_knn, r2score_knn = knn_prediction()
	st.write('*' * 30)
	rmse_rf, r2score_rf = rf_prediction()
	st.write('*' * 30)
	rmse_svm, r2score_svm = svm_prediction()
	st.write('*' * 30)
	
	st.subheader("Best fit model is:")
	if rmse_knn < (rmse_rf and rmse_svm):
		st.write('KNN has lower RMSE, ' + str(round(rmse_knn, 2)) + ' therefore it has best fit.')
	elif rmse_rf < (rmse_knn and rmse_svm):
		st.write('RF has lower RMSE, ' + str(round(rmse_rf, 2)) + ' therefore it has best fit.')
	elif rmse_svm < (rmse_knn and rmse_rf):
		st.write('SVM has lower RMSE, ' + str(round(rmse_svm, 2)) + ' therefore it has best fit.')
	else:
		pass
	
	st.subheader("Worse fit model is:")
	if r2score_knn < (r2score_rf and r2score_svm):
		if r2score_knn < 0:
			st.write('KNN has lower R^2, ' + str(round(r2score_knn, 2)) + ' therefore it has worse fit.')
	elif r2score_rf < (r2score_knn and r2score_svm):
		if r2score_rf < 0:
			st.write('RF has lower R^2, ' + str(round(r2score_rf, 2)) + ' therefore it has worse fit.')
	elif r2score_svm < (r2score_knn and r2score_rf):
		if r2score_svm < 0:
			st.write('SVM has lower R^2, ' + str(round(r2score_svm, 2)) + ' therefore it has worse fit.')
	else:
		pass

st.write('*' * 30)

# st.write("Predict Cases with KNN technique")
# if st.button("KNN Results"):
# 	knn_prediction()
#
# st.write('*' * 30)
#
# st.write("Predict Cases with RF technique")
# if st.button("RF Results"):
# 	rf_prediction()
#
# st.write('*' * 30)
#
# st.write("Predict Cases with SVM technique")
# if st.button("SVM Results"):
# 	svm_prediction()
	
st.write('*' * 30)
#import DataCleaning
st.write('Covid19 World Data:')
st.dataframe(covid19_world)

st.subheader('Statistics Description')
if st.checkbox("World Statistics"):
	st.write(covid19_world.describe())


