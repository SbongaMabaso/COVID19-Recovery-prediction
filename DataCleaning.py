from datetime import datetime

#print(datetime.now())

# for dataset handeling and calculations
import numpy as np
import pandas as pd
import requests

import json
from pandas.io.json import json_normalize
from bs4 import BeautifulSoup

from urllib.request import Request, urlopen


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

np.random.seed(42)
# tf.random.set_seed(42)

# Path to the file directory
path = os.getcwd()

if os.path.exists(path + "\data") == False:
	print("Make file directory!")
	os.mkdir(path + "\data")

mypath = path + "\data"
# path to the directory

os.chdir(mypath)

#print(os.getcwd())

# remove all '*.csv' files in the current directory
import glob

for file in glob.glob("*.csv"):
	os.remove(file)

#def data_update():
# read files from url
download_root = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'

confirmed_df = pd.read_csv(download_root + "time_series_covid19_confirmed_global.csv")
recovered_df = pd.read_csv(download_root + "time_series_covid19_recovered_global.csv")
deaths_df = pd.read_csv(download_root + "time_series_covid19_deaths_global.csv")

# save datasets as '*.csv' file
confirmed_df.to_csv('confirmed_df.csv', index=False)
recovered_df.to_csv('recovered_df.csv', index=False)
deaths_df.to_csv('deaths_df.csv', index=False)


confirmed_df.head(3)
deaths_df.head(3)
recovered_df.head()


# Merge a data sets  into one data main_df
# melt dataframes to go from wide to long
def melt_df(df, name):
	melted = pd.melt(df, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
	                 value_vars=confirmed_df.columns[4:],
	                 var_name='Date', value_name=name)
	return melted


melt_confirmed = melt_df(confirmed_df, 'Confirmed')
melt_deaths = melt_df(deaths_df, 'Deaths')
melt_recovered = melt_df(recovered_df, 'Recovered')

# merge dataframes
def merge(df1, df2):
	merged = pd.merge(left=df1, right=df2, how='left',
	                  on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long'])
	return merged


main_df = merge(merge(melt_confirmed, melt_deaths), melt_recovered)
main_df.to_csv('covid19_full_uncleaned.csv', index=False)

# Dataset preparation
# Percentage of missing data in each column

number_of_columns = main_df.shape[0]
percentage_of_missing_data = main_df.isnull().sum() / number_of_columns
# print(percentage_of_missing_data)

# fill missing values of 'Recovered' with 0
main_df['Recovered'] = main_df['Recovered'].fillna(0)

# fill missing values of 'province/state' with ''
main_df['Province/State'] = main_df['Province/State'].fillna('')

# convert values of 'Recovered' to int datatype
main_df['Recovered'] = main_df['Recovered'].astype('int')

# change names of some countries
main_df['Country/Region'] = main_df['Country/Region'].replace('Korea, South', "South Korea")
main_df['Country/Region'] = main_df['Country/Region'].replace('Mainland China', 'China')
main_df['Country/Region'] = main_df['Country/Region'].replace('Taiwan*', 'Taiwan')
main_df['Country/Region'] = main_df['Country/Region'].replace('Congo (Kinshasa)', 'Democratic Republic of the Congo')
main_df['Country/Region'] = main_df['Country/Region'].replace('Congo (Brazzaville)', 'Republic of the Congo')
main_df.loc[main_df['Province/State'] == 'Greenland', 'Country/Region'] = 'Greenland'

# add new column active cases for main_df
main_df['Active'] = main_df['Confirmed'] - main_df['Deaths'] - main_df['Recovered']
# print lat 5 rows
#print(main_df.tail())

# print(main_df['Date'].describe(), '\n')

# convert dates to proper date format for better visualization

main_df['Date'] = pd.to_datetime(main_df['Date']).dt.normalize()

# print(main_df['Date'].describe())
# print(main_df['Date'])

# remove rows in which 'Country/Region' is a ship name

main_df = main_df[main_df['Country/Region'].str.contains('Diamond Princess') != True]
main_df = main_df[main_df['Country/Region'].str.contains('MS Zaandam') != True]


main_df.to_csv('covid19_full_cleaned.csv', index=False)
main_df = pd.read_csv('covid19_full_cleaned.csv')
## Group covid19_full_cleaned dataframe by 'Date' and 'Country/Region'

covid19_date_country = main_df.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum() \
	.reset_index()

# add new columns containing new confirmed, new deaths and new recovered for each day
new_col = covid19_date_country.groupby(['Country/Region', 'Date'])['Confirmed', 'Deaths', 'Recovered'].sum() \
	.diff().reset_index()
# rename some columns of new_col
new_col.columns = ['Country/Region', 'Date', 'New confirmed', 'New deaths', 'New recovered']
columns = ['New confirmed', 'New deaths', 'New recovered']

# fix the value of first row of each country
first_rows = (new_col['Country/Region'] != new_col['Country/Region'].shift(1))
new_col.loc[first_rows, 'New confirmed'] = 0
new_col.loc[first_rows, 'New deaths'] = 0
new_col.loc[first_rows, 'New recovered'] = 0

# merging new values
covid19_date_country = pd.merge(covid19_date_country, new_col, on=['Country/Region', 'Date'])

# fill missing values with 0
covid19_date_country = covid19_date_country.fillna(0)

covid19_date_country['New confirmed'] = covid19_date_country['New confirmed'].apply(lambda x: 0 if x < 0 else x)

# fix datatypes of new columns
covid19_date_country[columns] = covid19_date_country[columns].astype('int')

#print(covid19_date_country.tail())

## Dataframe with the latest values of 'Country/Region'
# save as .csv file
covid19_date_country.to_csv('covid19_date_country.csv', index=False)

# save iran_df.csv file
iran_df = covid19_date_country[covid19_date_country['Country/Region'] == 'Iran']
iran_df.to_csv('covid19_iran.csv', index=False)

# get the latest values related to each country
covid19_country_latest = covid19_date_country[
	covid19_date_country['Date'] == max(covid19_date_country['Date'])].reset_index(drop=True) \
	.drop('Date', axis=1)

# print(covid19_country_latest.shape)
# print(covid19_country_latest['Country/Region'].unique().shape)

# mortality and recovery rates
covid19_country_latest['Recovery rate(per 100)'] = \
	np.round(100 * covid19_country_latest['Recovered'] / covid19_country_latest['Confirmed'], 2)
covid19_country_latest['Mortality rate(per 100)'] = \
	np.round(100 * covid19_country_latest['Deaths'] / covid19_country_latest['Confirmed'], 2)

# fill missing values with 0
columns = ['Recovery rate(per 100)', 'Mortality rate(per 100)']
covid19_country_latest[columns] = covid19_country_latest[columns].fillna(0)

#print(covid19_country_latest.head())
# save as .csv file
covid19_country_latest.to_csv('covid19_country_latest.csv', index=False)

## Group covid19_day_country dataframe by 'Date'
covid19_world = covid19_date_country.groupby('Date')[
	'Confirmed', 'Deaths', 'Recovered', 'Active', 'New confirmed', 'New deaths',
	'New recovered'].sum().reset_index()

# mortality and recovery rates

covid19_world['Recovery rate(per 100)'] = \
	np.round(100 * covid19_world['Recovered'] / covid19_world['Confirmed'], 2)
covid19_world['Mortality rate(per 100)'] = \
	np.round(100 * covid19_world['Deaths'] / covid19_world['Confirmed'], 2)

# Number of countries having non-zero confirmed cases in each date
covid19_world['Number of countries'] = covid19_date_country[covid19_date_country['Confirmed'] != 0].groupby('Date')[
	'Country/Region'] \
	.unique().apply(len).values

# fill missing values with 0
columns = ['Recovery rate(per 100)', 'Mortality rate(per 100)']
covid19_world[columns] = covid19_world[columns].fillna(0)

# st.success("Dataset updated")
# st.dataframe(covid19_world.head())

# save as '*.csv' file
covid19_world.to_csv('covid19_world.csv', index=False)

## *******************************************************************************************************************

covid19_country_latest.copy().sort_values('Confirmed', ascending=False) \
	.reset_index(drop=True).iloc[:30, :].style.bar(align='left', width=80, color='gold')

print(covid19_country_latest)

# import streamlit as st
# st.success("Dataset Updated Successful!")