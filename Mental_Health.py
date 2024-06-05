import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io

st.cache
# Read the CSV files

df_daly = pd.read_csv("Combined data_DALY.csv")
df_countries = pd.read_csv("Primary_Dim_Countries.csv")

# Convert to streamlit dataframes
st.dataframe(df_daly)
st.dataframe(df_countries)

#Display column names and data types
buffer = io.StringIO()
df_daly.info(buf=buffer)
s = buffer.getvalue()

st.text(s)

buffer = io.StringIO()
df_countries.info(buf=buffer)
s = buffer.getvalue()

st.text(s)
st.markdown(''' Get unique values and their count in the mental health legislation, policy and professional columns''')

st.write(df_daly.Mental_health_legislation.value_counts())
st.write(df_daly.Mental_health_policy.value_counts())
st.write(df_daly.Mental_health_professional.value_counts())


st.markdown('''The No data values are too high, so we'll drop these column''')
df_daly=df_daly.drop(columns=['Mental_health_legislation','Mental_health_policy','Mental_health_professional'], axis=1)

st.markdown('''Merge df_daly with df_countries''')

# Rename columns for merging
df_countries = df_countries.rename(columns={'Alpha-3 code': 'Country_Code'})
df_daly = df_daly.rename(columns={'Code (Data6)': 'Country_Code'})

# Merge the dataframes
merged_df=df_daly.merge(df_countries, on='Country_Code', how='left')
st.dataframe(merged_df)

st.markdown('We shall drop redundant columns from the data frame')
merged_df=merged_df.drop(columns=['Country_Code','Country Code','Country_Orig','Income group (group)'], axis=1)


#Display column names and data types
buffer = io.StringIO()
merged_df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)


st.subheader('Descriptive Statistics')
# Summary of the numercial columns
st.write(merged_df.describe())

# Summary of the categorical columns
st.write(merged_df.describe(include='object'))

st.subheader('Exploratory Data Analysis')
#check for null values
st.write(merged_df.isnull().sum())


#check the data for duplicates
st.write(merged_df.duplicated().sum())

st.markdown('**Outlier Detection**')

