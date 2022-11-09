# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:45:07 2022

@author: joshu
"""
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import tensorflow as tf


# Creating a function and loading the model
def COP_prediction(input_data):
    COP_model=tf.keras.models.load_model('C:/Users/joshu/Desktop/Streamlit/Keras ANN Model/COP/COP_model.h5')
    scaler_COP=pickle.load(open('C:/Users/joshu/Desktop/Streamlit/Keras ANN Model/COP/scaler_COP.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_COP.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    COP_modelprediction=COP_model.predict(input_data_reshaped)
    print(COP_modelprediction)
    return COP_modelprediction

def W1_prediction(input_data):
    W1_model=tf.keras.models.load_model('C:/Users/joshu/Desktop/Streamlit/Keras ANN Model/W1/W1_model.h5')
    scaler_W1=pickle.load(open('C:/Users/joshu/Desktop/Streamlit/Keras ANN Model/W1/scaler_W1.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_W1.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    W1_modelprediction=W1_model.predict(input_data_reshaped)
    print(W1_modelprediction)
    return W1_modelprediction

def W2_prediction(input_data):
    W2_model=tf.keras.models.load_model('C:/Users/joshu/Desktop/Streamlit/Keras ANN Model/W2/W2_model.h5')
    scaler_W2=pickle.load(open('C:/Users/joshu/Desktop/Streamlit/Keras ANN Model/W2/scaler_W2.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_W2.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    W2_modelprediction=W2_model.predict(input_data_reshaped)
    print(W2_modelprediction)
    return W2_modelprediction

def VapLoss_prediction(input_data):
    VapLoss_model=tf.keras.models.load_model('C:/Users/joshu/Desktop/Streamlit/Keras ANN Model/VapLoss/VapLoss_model.h5')
    scaler_VapLoss=pickle.load(open('C:/Users/joshu/Desktop/Streamlit/Keras ANN Model/VapLoss/scaler_VapLoss.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_VapLoss.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    VapLoss_modelprediction=VapLoss_model.predict(input_data_reshaped)
    print(VapLoss_modelprediction)
    return VapLoss_modelprediction


#Create title and slider
def main():
    # Giving a title
    st.title('2 Stage Refrigeration System')
    st.write('This app predicts the performance of a two stage refrigeration cycle')
    # Sidebar header
    st.sidebar.header('User Input Parameters')
    # Define user input features
    def user_input_features():
        RefrigerantFeed = st.sidebar.slider('Refrigerant Feed',5000,10000,5000)
        DP_LV9004 = st.sidebar.slider('Pressure drop across LV-9004',1140,1180,1140)
        DP_LV9005 = st.sidebar.slider('Pressure drop across LV-90045',290,340,290)
        CondenserDuty = st.sidebar.slider('Condenser Duty', 8.2, 8.6, 8.2)
        S12Ratio = st.sidebar.slider('Flow ratio of S12', 0.01, 0.02, 0.01, 0.0001,"%f")
        data = {'RefrigerantFeed': RefrigerantFeed,
                'DP_LV9004': DP_LV9004,
                'DP_LV9005': DP_LV9005,
                'CondenserDuty': CondenserDuty,
                'S12Ratio': S12Ratio}
        features = pd.DataFrame(data, index=[0])
        return features
# Create user input parameters title    
    df = user_input_features()
    st.subheader('User Input Parameters')
    st.write(df)
 
    
# Create subheaders for dependent variables
    col1, col2, col3 = st.columns(3)
    
    col1.subheader('Coefficient of Performance')
    result_COP = COP_prediction(df)
    series = pd.Series(result_COP[0])
    rounded_COP = round(series[0],3)
    col1.write(rounded_COP)

    
    col2.subheader('LP Compressor Duty (MW)')
    result_W1 = W1_prediction(df)
    series = pd.Series(result_W1[0])
    rounded_W1 = round(series[0],3)
    col2.write(rounded_W1)
    
    col3.subheader('HP Compressor Duty (MW)')
    result_W2 = W2_prediction(df)
    series = pd.Series(result_W2[0])
    rounded_W2 = round(series[0],3)
    col3.write(rounded_W2)
      
    
    col4, col5, col6 = st.columns(3)
    
    col4.subheader('Vapour Loss (kg/h)')
    result_VapLoss = VapLoss_prediction(df)
    series = pd.Series(result_VapLoss[0])
    rounded_VapLoss = round(series[0],3)
    col4.write(rounded_VapLoss) 




if __name__=='__main__':
    main()
    








    # OLD Code
    # Create subheaders for dependent variables
        #st.subheader('Coefficient of Performance')
        #result_COP = COP_prediction(df)
        #rounded = round(result[0],2)
        #st.write(result_COP)
    
    
    
    
    
    
    
    
    
    
   # st.subheader('Manual Input Section')
    #Getting the input data from the user 
    #RefrigerantFeed = st.number_input('Refrigerant Feed')
    #MolFractionPropane = st.number_input('Mol fraction of propane')
    #DP_LV9004 = st.number_input('Pressure drop across LV-9004')
    #DP_LV9005 = st.number_input('Pressure drop across LV-9005')
    #CondenserDuty = st.number_input('Condenser duty')
    #S12Ratio = st.number_input('Split fraction of S12')
    
    #output =''
    
    #creating a button for prediction
    #if st.button ('Predict the Coefficient of Performance'):
        #result = COP_prediction([[RefrigerantFeed,MolFractionPropane, DP_LV9004, DP_LV9005, CondenserDuty, S12Ratio]])
        #output = round(result[0],2)
    #st.success(output)
    

    
    
    
    
    
    
    
    
    