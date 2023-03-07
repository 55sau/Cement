import pandas as pd
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.title('CEMENT DEMAND FORECASTING')
uploaded_file = st.file_uploader(" ", type=['xlsx'])

if uploaded_file is not None:     
    data = pd.read_excel(uploaded_file)
    data['Date'] =data['Date'].apply(lambda x: x.strftime('%B-%Y'))
    hwe_model_mul_add = ExponentialSmoothing(data["Cement_Demand"][:1329], seasonal = "mul", trend = "add", seasonal_periods = 12).fit()
    
    newdata_pred = hwe_model_mul_add.predict(start = data.index[0], end = data.index[-1])
    
    st.subheader("EXPONENTIAL MODEL")
   
    st.write("DEMAND FORECASTING:", newdata_pred)
    
   
    
    
    
   
    
    

