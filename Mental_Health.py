import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import plotly.express as px

st.set_page_config(page_icon="Mental_Health_Analysis")
st.cache_data


# Read the CSV files
st.title('Global Mental Health Insights - Clustering Analysis')
st.markdown('''
            According to the World Health Organiztion (WHO), mental well being is not only the absence of mental illness but a state 
            of  well-being encompassing the physical, mental and social. The global burden of mental health disorders is significant and growing overtime.
            There is eunequal burden among countries and socio-economic groups. Interventions by key organizations, including WHO, play a crucial role in 
            supporting countries to strengthen their mental health systems and ensuring that mental health care is integrated to primary care settings.
            
            This project aims to explore:
            - The distribution and burden of mental health across different countries and regions
            - Examining WHO support in relation to the burden of mental disorders
            - Exploring the relationship between economic indicatiors and mental health indicators
            - Identify patterns and clusters of countries with similar mental health profiles, which can inform targeted interventions and policy recommendations

''')

st.markdown('**Key definitions**')
st.markdown('''
            - **DALY**: Disability-adjusted life year is a measure that represents the loss of the equivalent of one year of full health. 
                DALYs for a disease or health condition are the sum of years of life lost due to premature mortality (YLLs) and years of healthy life lost due 
                to disability (YLDs) due to prevalent cases of the disease or health condition in a population.
            - **Prevalence**: The proportion of a population who have a specific characteristic in a given time period.
            ''')

