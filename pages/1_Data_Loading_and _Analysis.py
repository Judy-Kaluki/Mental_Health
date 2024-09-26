import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import plotly.express as px

st.cache_data
st.subheader('Data loading and Analysis')

df_daly = pd.read_csv("Combined data_DALY.csv")
st.caption('Primary fact Da')
df_countries = pd.read_csv("Primary_Dim_Countries.csv")

# Convert to streamlit dataframes
st.dataframe(df_daly)
st.caption('Primary fact DALY')
st.dataframe(df_countries)
st.caption('Dim_Countries')


#Display column names and data types
st.markdown('**DALY Column names and data type**')
buffer = io.StringIO()
df_daly.info(buf=buffer)
s = buffer.getvalue()

st.text(s)

st.markdown('**Country Column names and data type**')
buffer = io.StringIO()
df_countries.info(buf=buffer)
s = buffer.getvalue()

st.text(s)

st.markdown('''**Unique column values and counts**''')

st.write(df_daly.Mental_health_legislation.value_counts())
st.write(df_daly.Mental_health_policy.value_counts())
st.write(df_daly.Mental_health_professional.value_counts())

st.markdown('**Merged data frame**')

# Rename columns for merging
df_countries = df_countries.rename(columns={'Alpha-3 code': 'Country_Code'})
df_daly = df_daly.rename(columns={'Code (Data6)': 'Country_Code'})

# Merge the dataframes
merged_df=df_daly.merge(df_countries, on='Country_Code', how='left')
st.dataframe(merged_df)

st.markdown('''We shall drop redundant columns from the data frame :Country_Code,Country Code,Income group (group)''')

merged_df=merged_df.drop(columns=['Country_Code','Country Code','Country_Orig','Income group (group)'], axis=1)

#Replace blank values in Income group with "No Region"
merged_df['Income group'].fillna("No Region", inplace=True)

st.markdown('**Merged_df Columns and data types**')
#Display column names and data types
buffer = io.StringIO()
merged_df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.subheader('Descriptive Statistics')
# Summary of the numercial columns
st.write(merged_df.describe())
st.markdown('''The DALY mean is 35,651 DALYs with a standard deviation of 197,589. The feature is widespread over a large range of values.
            The mental health policy, legislation and professional support are key factors contributing to DALY levels. However, majority of the rows are null. 
            Thus more data should be collected on these features to create a better understanding of the factors influencing DALYs.
'''
)

# Summary of the categorical columns
st.write(merged_df.describe(include='all'))

st.session_state['merged_df'] = merged_df
st.dataframe(merged_df)