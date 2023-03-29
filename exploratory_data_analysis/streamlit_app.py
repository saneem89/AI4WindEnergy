import streamlit as st
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

header = st.container()
col_select = st.container()
chart = st.container()
box = st.container()

@st.cache_data
def load_data():
    df11 = pd.read_csv('../data/processed/R80711.csv')
    df11['Date_time'] = pd.to_datetime(df11['Date_time'])

    df21 = pd.read_csv('../data/processed/R80721.csv')
    df21['Date_time'] = pd.to_datetime(df21['Date_time'])

    df36 = pd.read_csv('../data/processed/R80736.csv')
    df36['Date_time'] = pd.to_datetime(df36['Date_time'])

    df90 = pd.read_csv('../data/processed/R80790.csv')
    df90['Date_time'] = pd.to_datetime(df90['Date_time'])

    return df11, df21, df36, df90
df11, df21, df36, df90 = load_data()

data_description = pd.read_csv('../data/data_description.csv')

with header:
    st.title('Wind Turbines data exploration')

with col_select:
    st.header('Select two columns and date range')
    col_names = ['P_avg', 'Q_avg', 'Rm_avg', 'Rs_avg','Ya_avg', 'Yt_avg', 'Va_avg', 'Rbt_avg',  
                 'Ba_avg', 'Ws1_avg', 'Ws2_avg', 'Ws_avg', 'Wa_avg',  'Ot_avg', 'wind_speed', 
                 'wind_deg',  'temp', 'pressure','humidity', 'rain_1h', 'snow_1h']
    col1, col2, start, end = st.columns(4)
    with col1:
        data_col_name1 = st.selectbox('select column 1', options=col_names, index=2)
        col1_desc = data_description[data_description['Variable_name'] == data_col_name1]['Variable_long_name'].values[0]
        st.write(f'{col1_desc}')
    with col2:
        data_col_name2 = st.selectbox('select column 1', options=col_names, index=17)
        col2_desc = data_description[data_description['Variable_name'] == data_col_name2]['Variable_long_name'].values[0]
        st.write(f'{col2_desc}')
    with start:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01", format="%Y-%m-%d"))
    with end:
        end_date = st.date_input("End Date", value=pd.to_datetime("2015-03-01", format="%Y-%m-%d"))
    start_date = pd.Timestamp(start_date.strftime('%Y-%m-%d') + ' 00:00:00-01:00')
    end_date = pd.Timestamp(end_date.strftime('%Y-%m-%d') + ' 00:00:00-01:00')
    

with chart:
    st.header('Line Charts')
    df11_temp = df11[(df11['Date_time'] >= start_date) & (df11['Date_time'] <= end_date)]
    df21_temp = df21[(df21['Date_time'] >= start_date) & (df21['Date_time'] <= end_date)]
    df36_temp = df36[(df36['Date_time'] >= start_date) & (df36['Date_time'] <= end_date)]
    df90_temp = df90[(df90['Date_time'] >= start_date) & (df90['Date_time'] <= end_date)]
    
    fig, axis = plt.subplots(4, 2, figsize=(15, 20))
    plt.locator_params(axis='x', nbins=4)

    axis[0,0].set_title('R80711', fontsize=20)
    sns.lineplot(x='Date_time', y=data_col_name1, data=df11_temp, ax=axis[0,0])
    sns.lineplot(x='Date_time', y=data_col_name2, data=df11_temp, ax=axis[0,1])
    
    axis[1,0].set_title('R80721', fontsize=20)
    sns.lineplot(x='Date_time', y=data_col_name1, data=df21_temp, ax=axis[1,0])
    sns.lineplot(x='Date_time', y=data_col_name2, data=df21_temp, ax=axis[1,1])
    
    axis[2,0].set_title('R80736', fontsize=20)
    sns.lineplot(x='Date_time', y=data_col_name1, data=df36_temp, ax=axis[2,0])
    sns.lineplot(x='Date_time', y=data_col_name2, data=df36_temp, ax=axis[2,1])
    
    axis[3,0].set_title('R80790', fontsize=20)
    sns.lineplot(x='Date_time', y=data_col_name1, data=df90_temp, ax=axis[3,0])
    sns.lineplot(x='Date_time', y=data_col_name2, data=df90_temp, ax=axis[3,1])
    
    for i in range(4):
        axis[i,0].xaxis.set_major_locator(plt.MaxNLocator(5))
        axis[i,1].xaxis.set_major_locator(plt.MaxNLocator(5))
    st.pyplot(fig)

    st.subheader('Correlation matrix')

    figure = plt.figure(figsize=(10,8))
    sns.heatmap(df11_temp[col_names].corr())
    st.pyplot(figure)

with box:
    st.subheader('Box plot of selected columns')
    fig, axis = plt.subplots(4, 1, figsize=(10, 20))
    df11_temp[data_col_name2+'_bins'] = pd.cut(df11_temp[data_col_name2], 10)
    df21_temp[data_col_name2+'_bins'] = pd.cut(df21_temp[data_col_name2], 10)
    df36_temp[data_col_name2+'_bins'] = pd.cut(df36_temp[data_col_name2], 10)
    df90_temp[data_col_name2+'_bins'] = pd.cut(df90_temp[data_col_name2], 10)
    
    axis[0].set_title('R80711', fontsize=20)
    axis[1].set_title('R80721', fontsize=20)
    axis[2].set_title('R80736', fontsize=20)
    axis[3].set_title('R80790', fontsize=20)

    sns.boxplot(data=df11_temp, x=data_col_name2+'_bins', y=data_col_name1, ax=axis[0])
    sns.boxplot(data=df21_temp, x=data_col_name2+'_bins', y=data_col_name1, ax=axis[1])
    sns.boxplot(data=df36_temp, x=data_col_name2+'_bins', y=data_col_name1, ax=axis[2])
    sns.boxplot(data=df90_temp, x=data_col_name2+'_bins', y=data_col_name1, ax=axis[3])
    st.pyplot(fig)
    