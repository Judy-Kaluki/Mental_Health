import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Read the CSV files

df_daly=pd.read_csv("Primary_Fact_GlobalDALY_All.csv")
df_governance=pd.read_csv('Primary_Fact_GovernanceMentalHealth.csv')
df_spending=pd.read_csv('Primary_Fact_GovernmentSpendingMentalHealth.csv')
df_GDP=pd.read_csv("Primary_Fact_IMHE_GDP+HealthSpend.csv")
df_Age=pd.read_csv('Primary_Fact_MentalDisorder_Gender_Age_2000-17.csv')
df_population=pd.read_csv('Primary_Fact_Population_2000-17.csv')
df_support=pd.read_csv('Primary_Fact_WHOSupportServicesAvail.csv')
df_countries=pd.read_csv('Primary_Dim_Countries.csv')

st.cache_data


# Create a list of all DataFrame

dfs_All=[df_daly,df_governance,df_spending,df_GDP,df_Age,df_population,df_support,df_countries]

#List of DataFrame names
dfs_Names=['df_daly','df_governance','df_spending','df_GDP','df_Age','df_population','df_support','df_countries']

#Create Streamlit Data Frames

for df, df_name in zip(dfs_All, dfs_Names):
      st.subheader(f"Dataframe: {df_name}")
      st.dataframe(df)
#Given that the df_GDP is too large to display, we shall select specific columns for analysis before proceeding to merge the
#data to one df
st.subheader('Data Cleaning and Preprocessing')

df_daly=df_daly[['Cause','Code (Data6)','metric_name', 'sex_name', 'year (Data6)', 'val (Data6)']]
# filter to only include where the metric measure is percent
df_daly=df_daly.loc[df_daly['metric_name']=="Percent"]
# drop the metric_name column
daly_fil=df_daly.drop('metric_name', axis=1)

st.write(daly_fil.describe())


# Select necessary columns
df_governance=df_governance[['Code (Data)','Stand-alone mental health legislation','Stand-alone policy or plan for mental health', 'Year (Data)']]

df_GDP=df_GDP[["Code (Data5)", 'Metric','Year (Data5)','Health Expenditure per Capita (USD)']]

# Assumption that the expenditure refers to expenditure per capita, because there are instances where the metric name is GDP per capita
df_population=df_population[['Age Group', "Code", "Sex","Year","Population (#) - Female", "Population (#) - Male"]]

# Fill missing values in `Population (#) - Female` and `Population (#) - Male` with 0
df_population[['Population (#) - Female', 'Population (#) - Male']] = df_population[['Population (#) - Female', 'Population (#) - Male']].fillna(0)

# Add an additional column called Population, sex is identified under the sex column
df_population['Population'] = df_population['Population (#) - Female'] + df_population['Population (#) - Male']

# Select the necessary columns
df_population= df_population[['Population', 'Age Group', 'Code', 'Sex', 'Year']]

#Select the necessary columns and filter the data for only percent values
df_Age=df_Age[['age','cause','Code (Data 1+)', 'metric','sex','year','val']]
df_Age=df_Age.loc[df_Age['metric']=="Percent"]

#Create an updated list with the updated dfs then check for null values
dfs_Updated=[df_daly,df_governance,df_spending,df_GDP,df_Age,df_population,df_support,df_countries]
dfsNames_Updated=['df_daly','df_governance','df_spending','df_GDP','df_Age','df_population','df_support','df_countries']

#Create Streamlit Data Frames
for df, df_name in zip(dfs_Updated, dfsNames_Updated):
      st.subheader(f"Dataframe: {df_name}")
      st.dataframe(df)






#Loop through the df and display the first five rows

for df,df_name in (dfs_All, dfs_Names):
    print(f"\nFirst 5 rows of {df_name}:")
    print(df.head())

# df_daly: This file contains data on the global burden of mental health disease. A DALY is a measure of overall disease
#burden , expressed as the number of years lost due to disease, in this case mental disorders. The column represents the estimated 
#DALYs for each cause, entity and year: The dataser appears to cover various mental health conditions including: Anxiety disorders, ADHD,
# Bipolar, Depression, Eating disroders, schizophrenia 
# df_governance: the datasets indicates whethere there is a standalone policy for mental health
#df_spending: This indicates the government health spending on mental health
#df_GDP: The data set contains information on GDP per country: (explain GDP) and total helth spending

#Print the column names and data types for each

for df,df_name in (dfs_All, dfs_Names):
    print(f"\nColumn and data types of {df_name}:")
    print(df.info())

#Selecting the requisite columns


#Check for null values
for df,df_name in (dfs_All, dfs_Names):
    print(f"\n{df_name}:")
    print(df.isnull().sum())

#Check for duplicated values
#Check for null values
for df,df_name in (dfs_All, dfs_Names):
    print(f"\n{df_name}:")
    print(df.duplicated().sum())