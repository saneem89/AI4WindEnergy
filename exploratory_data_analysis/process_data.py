# Very basic processing of data in order to plot values.
# Covert date_time column to datetime format and add new columns for month, year and time of day.

import pandas as pd
import datetime
import matplotlib.pyplot as plt


def make_time_utc(row):
    return row['Date_time'].astimezone(datetime.timezone.utc)

# convert the date_time column to datetime format
def process_data(df):
    df['Date_time'] = pd.to_datetime(df['Date_time'])
    df['Date_time_utc'] = df.apply(make_time_utc, axis=1)
    df['month'] = pd.DatetimeIndex(df['Date_time_utc']).month
    df['year'] = pd.DatetimeIndex(df['Date_time_utc']).year
    df['time_of_day'] = pd.DatetimeIndex(df['Date_time_utc']).time
    return df

df11 = pd.read_csv('../data/R80711.csv')
df11 = process_data(df11)
df11.to_csv('../data/processed/R80711.csv')

df21 = pd.read_csv('../data/R80721.csv')
df21 = process_data(df21)
df21.to_csv('../data/processed/R80721.csv')


df36 = pd.read_csv('../data/R80736.csv')
df36 = process_data(df36)
df36.to_csv('../data/processed/R80736.csv')

df90 = pd.read_csv('../data/R80790.csv')
df90 = process_data(df90)
df90.to_csv('../data/processed/R80790.csv')