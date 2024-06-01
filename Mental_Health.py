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

st.markdown('**DALY DF**')
st.markdown('''From the DALY dataframe, the rows under the metric percent will be selected, as number and rate are different measures
            for the same observation. Then, rename the columns removing the spaces, and convert the year column to a date data type
''')

df_daly=df_daly[['Cause','Code (Data6)','metric_name', 'sex_name', 'year (Data6)', 'val (Data6)']]
# filter to only include where the metric measure is percent
df_daly=df_daly.loc[df_daly['metric_name']=="Percent"]
#rename the column names
df_daly_r=df_daly.rename(columns={'Code (Data6)': 'code', 'year (Data6)': 'year', 'val (Data6)': 'daly%','sex_name':'sex'})
#convert the year column to numeric values as non-numeric strings are contained in the data
#df_daly_r=pd.to_numeric(df_daly_r['year'].astype(str).str.replace(r'1/1/', '', regex=True))


st.markdown('**GOVERNANCE DF**')
# Select necessary columns
df_governance=df_governance[['Code (Data)','Stand-alone mental health legislation','Stand-alone policy or plan for mental health', 'Year (Data)']]
#rename the column names
df_governance=df_governance.rename(columns={'Code (Data)': 'code', 'Year (Data)': 'year','Stand-alone mental health legislation':'legislation','Stand-alone policy or plan for mental health':'policy'})
#convert the year column to numeric values
#df_governance=pd.to_numeric(df_governance['year'].astype(str).str.replace(r'1/1/', '', regex=True))

st.markdown('**GDP DF**')
df_GDP=df_GDP[["Code (Data5)", 'Metric','Year (Data5)','Health Expenditure per Capita (USD)']]
# rename the column names. Assumption that the expenditure refers to GDP per capita as well
df_GDP=df_GDP.rename(columns={"Code (Data5)": 'code', 'Year (Data5)': 'year','Health Expenditure per Capita (USD)':'expenditure'})
#convert the year column to numeric values
#df_GDP=pd.to_numeric(df_GDP['year'].astype(str).str.replace(r'1/1/', '', regex=True))

st.markdown('**Population DF**')
# Assumption that the expenditure refers to expenditure per capita, because there are instances where the metric name is GDP per capita
df_population=df_population[['Age Group', "Code", "Sex","Year","Population (#) - Female", "Population (#) - Male"]]
# Fill missing values in `Population (#) - Female` and `Population (#) - Male` with 0
df_population[['Population (#) - Female', 'Population (#) - Male']] = df_population[['Population (#) - Female', 'Population (#) - Male']].fillna(0)
# Add an additional column called Population, sex is identified under the sex column
df_population['Population'] = df_population['Population (#) - Female'] + df_population['Population (#) - Male']
# Select the necessary columns
df_population= df_population[['Population', 'Age Group', 'Code', 'Sex', 'Year']]
df_population=df_population.rename(columns={'Year':'year','Code':'code','Sex':'sex'})
#convert the year column to numeric values
#df_population=pd.to_numeric(df_population['year'].astype(str).str.replace(r'1/1/', '', regex=True))

st.markdown('**Age DF**')
#Select the necessary columns and filter the data for only percent values
df_Age=df_Age[['age','cause','Code (Data 1+)', 'metric','sex','year','val']]
#Select rows where metric=percent
df_Age=df_Age.loc[df_Age['metric']=="Percent"]
#Rename columns
df_Age=df_Age.rename(columns={'Code (Data 1+)':'code'})
#convert the year column to numeric values
#df_Age=pd.to_numeric(df_Age['year'].astype(str).str.replace(r'1/1/', '', regex=True))

st.markdown('**Spending, Support**')
#renaming the columns
df_spending=df_spending.rename({'Code (Data3)': 'code', 'Government Expenditure on Mental Health': 'expenditure_mental health'})

df_support=df_support.rename(columns={'Code (Data4)': 'code', 'Year (Data4)': 'year'})
#df_support=pd.to_numeric(df_support['year'].astype(str).str.replace(r'1/1/', '', regex=True))

#convert the year columns to date datatype
for df in[df_daly_r,df_governance,df_GDP,df_Age,df_support,df_population]:
     df['year']=pd.to_datetime(df['year'], format="%d/%m/%Y",errors='coerce').dt.year

#convert the year columns in population to date datatype

df_daly_r,df_governance,df_GDP,df_population,df_Age,df_spending,df_support,df_countries
#Create two data Frames (Focus on the disability adjusted life years)
merged_df=pd.merge(df_daly_r,df_countries, left_on='code', right_on='Alpha-3 code', how='left')
merged_df=pd.merge(merged_df,df_Age,on=['code','year','sex'], how='inner')
merged_df=pd.merge(merged_df,df_GDP,on=['code','year'], how='left')
merged_df=pd.merge(merged_df,df_governance,on=['code','year'], how='left')
merged_df=pd.merge(merged_df,df_spending,on=['code'], how='left')
merged_df=pd.merge(merged_df,df_support,on=['code'], how='left')

merged_df=pd.merge(merged_df,df_population,on=['code','year','sex'], how='inner')


st.dataframe(merged_df)

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