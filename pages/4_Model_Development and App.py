import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import plotly.express as px

merged_df = st.session_state.merged_df

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

