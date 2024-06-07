import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io

st.cache(suppress_st_warning=True)
# Read the CSV files
st.title('Global Mental Health Insights - Clustering Analysis')
st.markdown('''According to the World Health Organiztion (WHO), mental well being is not only the absence of mental illness but a state 
            of  well-being encompassing the physical, mental and social. The global burden of mental health disorders is significant and growing overtime.
            There is eunequal burden among countries and socio-economic groups. Interventions by key organizations, including WHO, play a crucial role in 
            supporting countries to strengthen their mental health systems and ensuring that mental health care is integrated to primary care settings.

            This project aims to explore:
            -The distribution and burden of mental health across different countries and regions
            -Examining WHO support in relation to the burden of mental disorders
            -Exploring the relationshio between economic indicatiors and mental health indicators
            -Indentify patterns and clusters of countries with similar mental health profiles, which can inform targeted interventions and policy recommendations

''')

st.markdown('**Key definitions**')
st.markdown('''-**DALY**: Disability-adjusted life year is a measure that represents the loss of the equivalent of one year of full health. 
                DALYs for a disease or health condition are the sum of years of life lost due to premature mortality (YLLs) and years of healthy life lost due 
                to disability (YLDs) due to prevalent cases of the disease or health condition in a population.
                
                -**Prevalence**: The proportion of a population who have a specific characteristic in a given time period.
            ''')

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
                            DALY_log=('DALY_log', 'mean'),
                            Health_exp_GDP=('Health_exp:GDP', 'mean'),
                            WHO_Support_100k=('WHO_Support per 100k','mean'),
                            Mental_health_professional=('Mental_health_professional','mean'))

st.dataframe(df_agg)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Drop rows where WHO support and mental health professional is zero
zero_df=df_agg[(df_agg['WHO_Support_100k']==0)|(df_agg['Mental_health_professional']==0)].index
df_agg=df_agg.drop(zero_df)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_agg.drop(['Country'], axis='columns'))

# Determine optimal number of clusters using the Elbow method
inertia = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Create a line plot of inertia vs. the number of clusters (Elbow method)
elbow_data = pd.DataFrame({'Clusters': range(2, 11), 'Inertia': inertia})
st.line_chart(elbow_data.set_index('Clusters'))

st.caption('Elbow method for optimal k')

# Apply K-Means clustering with the optimal number of clusters
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
df_agg['Cluster'] = kmeans.fit_predict(X_scaled)

# Display the first 5 rows of the aggregated dataframe with cluster assignments
df_agg['DALY']=np.exp(df_agg['DALY_log'])

st.dataframe(df_agg.head())

# Group by cluster and calculate statistics
cluster_stats = df_agg.groupby('Cluster')[['DALY', 'Health_exp_GDP', 'WHO_Support_100k','Mental_health_professional']].agg(['mean', 'std'])

st.dataframe(cluster_stats)

# Visual Display of the cluster
import plotly.express as px
fig_plotly = px.scatter(
    df_agg, 
    x="Health_exp_GDP", 
    y="DALY", 
    color="Cluster", 
    size="Mental_health_professional",
    hover_data=['Country'],
)
st.caption('Cluster of mental health burden')

# Customize Plotly plot layout
fig_plotly.update_layout(
    xaxis_title="Average ratio of health_expenditure to GDP",
    yaxis_title="DALYs",
    legend_title_text='Cluster'
)

# Display Plotly plot in Streamlit
st.plotly_chart(fig_plotly)
st.caption('Cluster of countries')

st.caption('Cluster of mental health burden')

st.subheader('Cluster App')

st.sidebar.header('Cluster Selection')
selected_cluster = st.sidebar.selectbox('Select a Cluster to View', df_agg['Cluster'].unique())
# Filter data for selected cluster
filtered_df = df_agg[df_agg['Cluster'] == selected_cluster]

# Display scatter plot of selected cluster
st.markdown(f'**Scatter Plot of Cluster {selected_cluster}**')

fig_plotly = px.scatter(
    filtered_df, 
    x="Health_exp_GDP", 
    y="DALY", 
    color='Income group',
    size="Mental_health_professional",
    hover_data=['Country'],
)
# Customize Plotly plot layout
fig_plotly.update_layout(
    xaxis_title="Average ratio of health_expenditure to GDP",
    yaxis_title="DALYs",
    legend_title_text='Cluster'
)
st.plotly_chart(fig_plotly)
st.caption(f"Scatter Plot: Cluster {selected_cluster}")

# Display cluster statistics
st.markdown(f'**Cluster {selected_cluster} Statistics**')
st.table(cluster_stats.loc[selected_cluster])

# Display list of countries in selected cluster
st.markdown(f'**Countries in Cluster {selected_cluster}**')
st.table(filtered_df['Country'])