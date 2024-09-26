import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import plotly.express as px


merged_df = st.session_state.merged_df

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

st.session_state['merged_df'] = merged_df