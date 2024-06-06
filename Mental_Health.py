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
#df_daly=df_daly.drop(columns=['Mental_health_legislation','Mental_health_policy','Mental_health_professional'], axis=1)

st.markdown('''Merge df_daly with df_countries''')

# Rename columns for merging
df_countries = df_countries.rename(columns={'Alpha-3 code': 'Country_Code'})
df_daly = df_daly.rename(columns={'Code (Data6)': 'Country_Code'})

# Merge the dataframes
merged_df=df_daly.merge(df_countries, on='Country_Code', how='left')
st.dataframe(merged_df)

st.markdown('We shall drop redundant columns from the data frame')
merged_df=merged_df.drop(columns=['Country_Code','Country Code','Country_Orig','Income group (group)'], axis=1)

#Replace blank values in Income group with "No Region"
merged_df['Income group'].fillna("No Region", inplace=True)

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
# Filter numerical columns for selection
numerical_columns = ['DALY', 'GDP_per_Capita', 'WHO_Support per 100k', 'Health expenditure per capita.Value']

# Streamlit selectbox for variable selection
st.subheader("Variable Selection")
selected_variable = st.selectbox("Select a numerical variable:", numerical_columns)

# Box Plot
st.subheader(f"Box Plot of {selected_variable}")
fig, ax = plt.subplots()
sns.boxplot(data=merged_df, y=selected_variable, ax=ax)
ax.set_title(f"Box Plot of {selected_variable}")
st.pyplot(fig)

# Histogram
st.subheader(f"Histogram of {selected_variable}")
fig, ax = plt.subplots()
sns.histplot(data=merged_df, x=selected_variable, bins=30, kde=True, ax=ax)
ax.set_title(f"Histogram of {selected_variable}")
st.pyplot(fig)

st.markdown('''From the above we can note that the there is a considerable spread in the DALY variable and numerous 
            outliers which shows that many countries have DALY values that are considerably higher than the average.Further, from the 
            histogram, we can observe that the data is positively skewed with a long tail. This means that a majority of the country
            are concentrated around the lower end of the DALY distribution
            
            A similar trend is observed with the GDP per capita and health expenditure per capita values. From the above we can summarize the 
            following
            - From the high DALY values, there is a significant burden of mental health disorders globally
            - From the distribution, some countries experience a disproportionately higher burden as compared to others
            - 
            '''
    
)
st.subheader(''' Feature engineering
'''
)
st.markdown('**Normalization**')
st.markdown('''As observed the DALY column is positively skewed. We shall apply a log transformation to normalize the data
            
            ''')
# As the data set contains zero values, we shall add a small constant of 1 before applying the transformation
# Apply log transformation to DALY after adding 1
merged_df['DALY_log'] = merged_df['DALY'].apply(lambda x: np.log(x + 1))

# Histogram of the log transformed values
fig, ax = plt.subplots()
sns.histplot(data=merged_df, x='DALY_log', kde=True, ax=ax)
ax.set_title(f"Histogram of DALY_log")
st.pyplot(fig)
st.caption('Histogram after log transformation')

st.markdown('**Additional Variables**')
#Add a column- ratio of health expenditure to GDP
merged_df['Health_exp:GDP']=(merged_df['Health expenditure per capita.Value']/merged_df['GDP_per_Capita']).replace([np.inf, -np.inf], 0)

st. markdown('**Correlation analysis')
st.markdown('''To analyze the correlation, the categorical columns require encoding to a numerical format
            ''')

from sklearn.preprocessing import LabelEncoder
# Filter categorical columns
categorical_cols = merged_df.select_dtypes(include=['object']).columns

# Initialize LabelEncoder
le = LabelEncoder()

# Apply label encoding to each categorical column
for col in categorical_cols:
    merged_df[col] = le.fit_transform(merged_df[col])

# Calculate the correlation matrix
correlation_matrix = merged_df.corr()

#Visualize the correlation matrix
fig, ax = plt.subplots()
sns.heatmap(merged_df.corr(), ax=ax)
st.pyplot(fig)
st.caption('Correlation matrix')

#Display the correlation matrix in a table
st.write(correlation_matrix.style.background_gradient(cmap='coolwarm'))

st.markdown('''- DALY and GDP per capita are negatively correlated. This suggests that wealthier countries haver 
            lower DALY values

               - DALY and health expenditure per capita are positively correlated. This suggest that countries
            with high health expenditure also have high values. This is the impact of high burden of mental disease, as countries spend more
            to address these health concerns 


            ''')

st.subheader('Feature Selection')
st.markdown(''' To select features, we shall assess feature importance. As the target variable (DALY) is continuous,
            we shall utilize a regression model- Random Forest Regressor. To reduce potential for data leakage we shall exclude country, region and 
            cause from the selection.
''')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Select features for X
merged_df=merged_df.dropna()

X = merged_df.drop((['DALY_log','DALY','Country','Region','Cause']), axis='columns')
# Select target variable for y
y = merged_df['DALY_log']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame to store feature importances
feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

#Visualize the feature importances
fig, ax = plt.subplots()
sns.barplot(feature_importances_df, x='Feature',y='Importance')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)
st.caption('Feature importances')



st.markdown(''' From the Analysis, the economic indicators play a primary role in influencing DALY values
            Year also also plays a role, indicating changing DALY values over time. 
''')

st.subheader('Model development')
#Aggregate the dataframe for clustering

df_agg=merged_df.groupby(['Country','Income group','Year'], as_index=False).agg(
                            DALY_log=('DALY_log', 'median'),
                            Health_exp_GDP=('Health_exp:GDP', 'mean'),
                            WHO_Support_100k=('WHO_Support per 100k','mean'),
                            Mental_health_professional=('Mental_health_professional','mean'))

st.dataframe(df_agg)