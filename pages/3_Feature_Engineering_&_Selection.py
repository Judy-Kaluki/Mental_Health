import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import plotly.express as px


merged_df = st.session_state.merged_df

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

st.markdown(' We shall add a new variable health expenditure:GDP ratio to determine the share of GDP going to health')


#Add a column- ratio of health expenditure to GDP
merged_df['Health_exp:GDP']=(merged_df['Health expenditure per capita.Value']/merged_df['GDP_per_Capita']).replace([np.inf, -np.inf], 0)

st. markdown('**Correlation analysis**')
st.markdown('''To analyze the correlation, the categorical columns require encoding to a numerical format
            ''')

from sklearn.preprocessing import LabelEncoder
# Filter categorical columns
st.write(merged_df.columns)

categorical_cols = merged_df.drop(['Country'], axis='columns').select_dtypes(include=['object']).columns

# Initialize LabelEncoder
le = LabelEncoder()

# Apply label encoding to each categorical column
for col in categorical_cols:
    merged_df[col] = le.fit_transform(merged_df[col])

# Calculate the correlation matrix
correlation_matrix = merged_df.drop(['Country'], axis='columns').corr()

#Visualize the correlation matrix
fig, ax = plt.subplots()
sns.heatmap(merged_df.drop(['Country'], axis='columns').corr(), ax=ax)
st.pyplot(fig)
st.caption('Correlation matrix')

#Display the correlation matrix in a table
st.write(correlation_matrix.style.background_gradient(cmap='coolwarm'))

st.markdown('''
            - DALY and GDP per capita are negatively correlated. This suggests that high income countries have lower DALY values
            than lower income countries. 
            - DALY and health expenditure per capita, and the share of GDP going to health expenditure are positively correlated.
            This presents the impact of mental health disorders, where higher expenditure is directed towards addressing the indentified health concerns.
            Further, with more funding directed towards health, more cases can be identified and treated.  
            -WHO support has increased over time as well as more countries adopting mental health policy and legislation 

            ''')

st.subheader('Feature Selection')
st.markdown(''' To select features, we shall assess feature importance. As the target variable (DALY) is continuous,
            we shall utilize a regression model- Random Forest Regressor. To reduce potential for data leakage we shall exclude country, region, 
            cause and health:exp ratio from the selection.
''')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Select features for X
merged_df=merged_df.dropna()

X = merged_df.drop((['DALY_log','DALY','Country','Region','Cause','Health_exp:GDP']), axis='columns')
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

st.markdown(''' 
                It is clear that the economic indicators of health expenditure per capita and GDP per capita heavily influence 
                DALY values. This supports the earlier observations of higher funding towards healthcare promoted indentification
            and addressing of mental health disorders.
                Further, DALY values have been increasing over time due to increased awareness, increased identification of cases.

''')

st.session_state['merged_df'] = merged_df