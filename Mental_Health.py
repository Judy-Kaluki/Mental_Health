import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import plotly.express as px

st.cache(suppress_st_warning=True)
# Read the CSV files
st.title('Global Mental Health Insights - Clustering Analysis')
st.markdown('''
            According to the World Health Organiztion (WHO), mental well being is not only the absence of mental illness but a state 
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
st.markdown('''
            -**DALY**: Disability-adjusted life year is a measure that represents the loss of the equivalent of one year of full health. 
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

st.subheader('Exploratory Data Analysis')
#check for null values
st.markdown('**Check for null values**')
st.write(merged_df.isnull().sum())
st.markdown('Check for duplicates')
#check the data for duplicates
st.write(merged_df.duplicated().sum())

st.markdown('**Outlier Detection**')
# Filter numerical columns for selection
numerical_columns = ['DALY', 'GDP_per_Capita', 'WHO_Support per 100.k', 'Health expenditure per capita.Value']

# Streamlit selectbox for variable selection
st.markdown("**Variable Selection**")
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

st.markdown('''
            
            From the above we can note that the there is a considerable spread in the DALY variable and numerous 
            outliers. The data shows that many countries have DALY values that are considerably higher than the average.Further, from the 
            histogram, we can observe that the data is positively skewed with a long tail. This means that a majority of the country
            are concentrated around the lower end of the DALY distribution. A similar trend is observed with the GDP per capita and health expenditure per capita values. 
            Insights from the plots can be summarised as follows
                - The high DALY values signify a heavy burden of mental health disorders globally
                - From the distribution, some countries experience a disproportionately higher burden as compared to others
                - The median WHO support per 100,000 is 1.9 amd majority of the countries are on the lower end of the distribution.
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

st.markdown(' We shall add a new variable health expenditure:GDP ratio to determine the share of GDP going to health')


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

st.subheader('Model development')
#Aggregate the dataframe for clustering
st.markdown('''
            The aim of the project is to group countries share similar mental health profiles, which can inform targeted interventions
            and advise on national mental health policies. Further, strategies can be specificaly tailored to address the different clusters

''')

df_agg=merged_df.groupby(['Country','Income group','Year'], as_index=False).agg(
                            DALY_log=('DALY_log', 'mean'),
                            Health_exp_GDP=('Health_exp:GDP', 'mean'),
                            WHO_Support_100k=('WHO_Support per 100k','mean'),
                            Mental_health_professional=('Mental_health_professional','mean'))

st.dataframe(df_agg)
st.caption('Dataframe aggregated by country, income group and year')

st.markdown('**Clustering using K-means**')
st.markdown('''
            To model the clusters, K-means algorithm shall be used. As one of the assumptions on the data 
            is that the variance within each cluster is roughly equal across all dimensions, the features are standardized.
            Thes elbow method was used to determine the optimal number of clusters.
'''
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Drop rows where WHO support or mental health professional is zero
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

st.markdown('''
            From the chart above, the optimal number of clusters is 3

'''

)

# Apply K-Means clustering with the optimal number of clusters
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
df_agg['Cluster'] = kmeans.fit_predict(X_scaled)

# Display the first 5 rows of the aggregated dataframe with cluster assignments
df_agg['DALY']=np.exp(df_agg['DALY_log'])

st.markdown('**Cluster Analysis')
st.markdown('''
            Dataframe with the assigned clusters is shown below.
''')
st.dataframe(df_agg.head())

st.markdown('''
            The three different and distinct clusters are shown below:
            - Cluster 0: This is the cluster with the highest DALY at a mean DALY of 37,914. Further, it has the lowest health expenditure
            to GDP ratio and the lowest WHO support per 100k. For this cluster, increasing the WHO support
            and funding health care is more likely to improve the mental health outcomes
            - Cluster 1: This cluster has moderate average DALYs, with the highest WHO support_100k. However these countries
            have the lowest average mental health professionals. They also have the highest healthcare funding in relation to GDP per 
            capita. The countries could prioritize reallocation of resources to train more mental health professional, development
            of mental health facilities
            - Cluster 2: These are the countries with the lowest average DALYs and the highest average mental health professional per 100,000.
            These countries have relatively better mental health outcomes and resource allocation as compared to the other clusters.
            ''' 

)
# Group by cluster and calculate statistics
cluster_stats = df_agg.groupby('Cluster')[['DALY', 'Health_exp_GDP', 'WHO_Support_100k','Mental_health_professional']].agg(['mean', 'std'])

st.dataframe(cluster_stats)

# Visual Display of the cluster

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

