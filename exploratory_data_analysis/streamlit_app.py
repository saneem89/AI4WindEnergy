import streamlit as st
import pandas as pd

header = st.container()
col_select = st.container()
chart = st.container()

df11 = pd.read_csv('../data/R80711.csv')
df21 = pd.read_csv('../data/R80721.csv')
df36 = pd.read_csv('../data/R80736.csv')
df90 = pd.read_csv('../data/R80790.csv')


with header:
    st.title('Data Exploration')

with col_select:
    st.subheader('Plots')
    col_names = list(df11.columns)
    col_names.pop(0); col_names.pop(0) # removing first two columns (Date_time and Date_time_rn)
    data_col_name = st.selectbox('Select a column', options=col_names)

with chart:
    st.subheader('Chart')


    st.line_chart(pd.DataFrame(df11[['Date_time', data_col_name]]))