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

primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

# Creating a function and loading the model
def COP_prediction(input_data):
    COP_model=pickle.load(open('COP_model.sav','rb'))
    scaler_COP=pickle.load(open('scaler_COP.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_COP.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    COP_modelprediction=COP_model.predict(input_data_reshaped)
    print(COP_modelprediction)
    return COP_modelprediction

def W1_prediction(input_data):
    W1_model=pickle.load(open('W1_model.sav','rb'))
    scaler_W1=pickle.load(open('scaler_W1.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_W1.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    W1_modelprediction=W1_model.predict(input_data_reshaped)
    print(W1_modelprediction)
    return W1_modelprediction

def W2_prediction(input_data):
    W2_model=pickle.load(open('W2_model.sav','rb'))
    scaler_W2=pickle.load(open('scaler_W2.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_W2.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    W2_modelprediction=W2_model.predict(input_data_reshaped)
    print(W2_modelprediction)
    return W2_modelprediction

def VapLoss_prediction(input_data):
    VapLoss_model=tf.keras.models.load_model('VapLoss_model.h5')
    scaler_VapLoss=pickle.load(open('scaler_VapLoss.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_VapLoss.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    VapLoss_modelprediction=VapLoss_model.predict(input_data_reshaped)
    print(VapLoss_modelprediction)
    return VapLoss_modelprediction

def LiqLoss_prediction(input_data):
    LiqLoss_model=tf.keras.models.load_model('LiqLoss_model.h5')
    scaler_LiqLoss=pickle.load(open('scaler_LiqLoss.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_LiqLoss.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    LiqLoss_modelprediction=LiqLoss_model.predict(input_data_reshaped)
    print(LiqLoss_modelprediction)
    return LiqLoss_modelprediction

def D0001T_prediction(input_data):
    D0001T_model=pickle.load(open('D0001T_model.sav','rb'))
    scaler_D0001T=pickle.load(open('scaler_D0001T.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_D0001T.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    D0001T_modelprediction=D0001T_model.predict(input_data_reshaped)
    print(D0001T_modelprediction)
    return D0001T_modelprediction

def D0002T_prediction(input_data):
    D0002T_model=pickle.load(open('D0002T_model.sav','rb'))
    scaler_D0002T=pickle.load(open('scaler_D0002T.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_D0002T.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    D0002T_modelprediction=D0002T_model.predict(input_data_reshaped)
    print(D0002T_modelprediction)
    return D0002T_modelprediction

def D0003T_prediction(input_data):
    D0003T_model=pickle.load(open('D0003T_model.sav','rb'))
    scaler_D0003T=pickle.load(open('scaler_D0003T.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_D0003T.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    D0003T_modelprediction=D0003T_model.predict(input_data_reshaped)
    print(D0003T_modelprediction)
    return D0003T_modelprediction

def D0004T_prediction(input_data):
    D0004T_model=pickle.load(open('D0004T_model.sav','rb'))
    scaler_D0004T=pickle.load(open('scaler_D0004T.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_D0004T.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    D0004T_modelprediction=D0004T_model.predict(input_data_reshaped)
    print(D0004T_modelprediction)
    return D0004T_modelprediction

def D0005T_prediction(input_data):
    D0005T_model=pickle.load(open('D0005T_model.sav','rb'))
    scaler_D0005T=pickle.load(open('scaler_D0005T.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_D0005T.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    D0005T_modelprediction=D0005T_model.predict(input_data_reshaped)
    print(D0005T_modelprediction)
    return D0005T_modelprediction

def D0001P_prediction(input_data):
    D0001P_model=pickle.load(open('D0001P_model.sav','rb'))
    scaler_D0001P=pickle.load(open('scaler_D0001P.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_D0001P.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    D0001P_modelprediction=D0001P_model.predict(input_data_reshaped)
    print(D0001P_modelprediction)
    return D0001P_modelprediction

def D0002P_prediction(input_data):
    D0002P_model=pickle.load(open('D0002P_model.sav','rb'))
    scaler_D0002P=pickle.load(open('scaler_D0002P.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_D0002P.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    D0002P_modelprediction=D0002P_model.predict(input_data_reshaped)
    print(D0002P_modelprediction)
    return D0002P_modelprediction

def D0003P_prediction(input_data):
    D0003P_model=pickle.load(open('D0003P_model.sav','rb'))
    scaler_D0003P=pickle.load(open('scaler_D0003P.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_D0003P.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    D0003P_modelprediction=D0003P_model.predict(input_data_reshaped)
    print(D0003P_modelprediction)
    return D0003P_modelprediction

def D0004P_prediction(input_data):
    D0004P_model=pickle.load(open('D0004P_model.sav','rb'))
    scaler_D0004P=pickle.load(open('scaler_D0004P.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_D0004P.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    D0004P_modelprediction=D0004P_model.predict(input_data_reshaped)
    print(D0004P_modelprediction)
    return D0004P_modelprediction

def D0005P_prediction(input_data):
    D0005P_model=pickle.load(open('D0005P_model.sav','rb'))
    scaler_D0005P=pickle.load(open('scaler_D0005P.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_D0005P.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    D0005P_modelprediction=D0005P_model.predict(input_data_reshaped)
    print(D0005P_modelprediction)
    return D0005P_modelprediction

def EB001_prediction(input_data):
    EB001_model=pickle.load(open('EB001_model.sav','rb'))
    scaler_EB001=pickle.load(open('scaler_EB001.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_EB001.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    EB001_modelprediction=EB001_model.predict(input_data_reshaped)
    print(EB001_modelprediction)
    return EB001_modelprediction

def EB002_prediction(input_data):
    EB002_model=pickle.load(open('EB002_model.sav','rb'))
    scaler_EB002=pickle.load(open('scaler_EB002.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_EB002.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    EB002_modelprediction=EB002_model.predict(input_data_reshaped)
    print(EB002_modelprediction)
    return EB002_modelprediction

def EB003_prediction(input_data):
    EB003_model=pickle.load(open('EB003_model.sav','rb'))
    scaler_EB003=pickle.load(open('scaler_EB003.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_EB003.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    EB003_modelprediction=EB003_model.predict(input_data_reshaped)
    print(EB003_modelprediction)
    return EB003_modelprediction

def EB004_prediction(input_data):
    EB004_model=pickle.load(open('EB004_model.sav','rb'))
    scaler_EB004=pickle.load(open('scaler_EB004.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_EB004.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    EB004_modelprediction=EB004_model.predict(input_data_reshaped)
    print(EB004_modelprediction)
    return EB004_modelprediction

def EB005_prediction(input_data):
    EB005_model=pickle.load(open('EB005_model.sav','rb'))
    scaler_EB005=pickle.load(open('scaler_EB005.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_EB005.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    EB005_modelprediction=EB005_model.predict(input_data_reshaped)
    print(EB005_modelprediction)
    return EB005_modelprediction

def Condenser_prediction(input_data):
    Condenser_model=pickle.load(open('Condenser_model.sav','rb'))
    scaler_Condenser=pickle.load(open('scaler_Condenser.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_Condenser.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    Condenser_modelprediction=Condenser_model.predict(input_data_reshaped)
    print(Condenser_modelprediction)
    return Condenser_modelprediction

def VF_prediction(input_data):
    VF_model=pickle.load(open('VF_model.sav','rb'))
    scaler_VF=pickle.load(open('scaler_VF.sav','rb'))
    input_data_as_numpy_array=np.asarray(input_data)
    std_data=scaler_VF.transform(input_data_as_numpy_array)
    input_data_reshaped=std_data.reshape(1,-1)
    VF_modelprediction=VF_model.predict(input_data_reshaped)
    print(VF_modelprediction)
    return VF_modelprediction


st.title('Two-Stage Refrigeration System')
st.text('This web app predicts the performance of a two-stage refrigeration plant.')

tab1, tab2, tab3 = st.tabs(["Performance data", "Process", "About"])
########################################################################
#Create title and slider
def main():
    # Sidebar header
    # Define user input features
 with tab1:
    st.sidebar.header('User Input Parameters')
    def user_input_features():
        RefrigerantFeed = st.sidebar.slider('Refrigerant Feed',5000,10000,7500, 1, "%f" )
        DP_LV9004 = st.sidebar.slider('Pressure drop across LV-9004',1140,1180,1160)
        DP_LV9005 = st.sidebar.slider('Pressure drop across LV-9005',290,340,310)
        CondenserDuty = st.sidebar.slider('Condenser Duty', 8.2, 8.6, 8.4)
        S12Ratio = st.sidebar.slider('Flow ratio of S12', 0.01, 0.025, 0.017, 0.0001,"%f")
        data = {'RefrigerantFeed': RefrigerantFeed,
                'DP_LV9004': DP_LV9004,
                'DP_LV9005': DP_LV9005,
                'CondenserDuty': CondenserDuty,
                'S12Ratio': S12Ratio}
        features = pd.DataFrame(data, index=[0])
        return features
# Create user input parameters title    
    df = user_input_features()
    #st.subheader('User Input Parameters')
    #st.write(df)
    
########################################################################
# Create subheaders for main performance indicator  
    new_title = '<p style="font-family:monospace; color:red; font-size: 30px;">Main Performance Indicator</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.text('This section displays the main performance indicators of the refrigeration system.')
    
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
    rounded_VapLoss = round(series[0],2)
    col4.write(rounded_VapLoss) 
    
    col5.subheader('Liquid Loss (kg/h)')
    result_LiqLoss = LiqLoss_prediction(df)
    series = pd.Series(result_LiqLoss[0])
    rounded_LiqLoss = round(series[0],2)
    col5.write(rounded_LiqLoss)

########################################################################
# Create subheaders for vessel operating conditions
    new_title = '<p style="font-family:monospace; color:red; font-size: 30px;">Vessel Operating Conditions</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.text('This section displays the operating temperature and pressure of the vessels.')
    col7, col8, col9, col10, col11 = st.columns(5)
    
    col7.subheader('D-0001 (°C)')
    result_D0001T = D0001T_prediction(df)
    series = pd.Series(result_D0001T[0])
    rounded_D0001T = round(series[0],3)
    col7.write(rounded_D0001T)

    col8.subheader('D-0002 (°C)')
    result_D0002T = D0002T_prediction(df)
    series = pd.Series(result_D0002T[0])
    rounded_D0002T = round(series[0],3)
    col8.write(rounded_D0002T)
    
    col9.subheader('D-0003 (°C)')
    result_D0003T = D0003T_prediction(df)
    series = pd.Series(result_D0003T[0])
    rounded_D0003T = round(series[0],3)
    col9.write(rounded_D0003T)
    
    col10.subheader('D-0004 (°C)')
    result_D0004T = D0004T_prediction(df)
    series = pd.Series(result_D0004T[0])
    rounded_D0004T = round(series[0],3)
    col10.write(rounded_D0004T)

    col11.subheader('D-0005 (°C)')
    result_D0005T = D0005T_prediction(df)
    series = pd.Series(result_D0005T[0])
    rounded_D0005T = round(series[0],3)
    col11.write(rounded_D0005T)

    col12, col13, col14, col15, col16 = st.columns(5)
    col12.subheader('D-0001 (kPag)')
    result_D0001P = D0001P_prediction(df)
    series = pd.Series(result_D0001P[0])
    rounded_D0001P = round(series[0],3)
    col12.write(rounded_D0001P)
 
    col13.subheader('D-0002 (kPag)')
    result_D0002P = D0002P_prediction(df)
    series = pd.Series(result_D0002P[0])
    rounded_D0002P = round(series[0],3)
    col13.write(rounded_D0002P)

    col14.subheader('D-0003 (kPag)')
    result_D0003P = D0003P_prediction(df)
    series = pd.Series(result_D0003P[0])
    rounded_D0003P = round(series[0],3)
    col14.write(rounded_D0003P)

    col15.subheader('D-0004 (kPag)')
    result_D0004P = D0004P_prediction(df)
    series = pd.Series(result_D0004P[0])
    rounded_D0004P = round(series[0],3)
    col15.write(rounded_D0004P)

    col16.subheader('D-0005 (kPag)')
    result_D0005P = D0005P_prediction(df)
    series = pd.Series(result_D0005P[0])
    rounded_D0005P = round(series[0],3)
    col16.write(rounded_D0005P)

########################################################################
# Create subheaders for flow through evaporators
    new_title = '<p style="font-family:monospace; color:red; font-size: 30px;">Mass Flowrate to Evaporators</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.text('This section displays the mass flow through all evaporators in the system.')

    col17, col18, col19, col20, col21 = st.columns(5)
    
    col17.subheader('EB-001 (kg/h)')
    result_EB001 = EB001_prediction(df)
    series = pd.Series(result_EB001[0])
    rounded_EB001 = round(series[0],2)
    col17.write(rounded_EB001)

    col18.subheader('EB-002 (kg/h)')
    result_EB002 = EB002_prediction(df)
    series = pd.Series(result_EB002[0])
    rounded_EB002 = round(series[0],2)
    col18.write(rounded_EB002)

    col19.subheader('EB-003 (kg/h)')
    result_EB003 = EB003_prediction(df)
    series = pd.Series(result_EB003[0])
    rounded_EB003 = round(series[0],2)
    col19.write(rounded_EB003)

    col20.subheader('EB-004 (kg/h)')
    result_EB004 = EB004_prediction(df)
    series = pd.Series(result_EB004[0])
    rounded_EB004 = round(series[0],2)
    col20.write(rounded_EB004)

    col21.subheader('EB-005 (kg/h)')
    result_EB005 = EB005_prediction(df)
    series = pd.Series(result_EB005[0])
    rounded_EB005 = round(series[0],2)
    col21.write(rounded_EB005)

########################################################################
# Create subheader for condenser
    new_title = '<p style="font-family:monospace; color:red; font-size: 30px;">Condenser Outlet</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.text('This section displays the mass flow and vapour fraction at the condenser outlet.')

    col22, col23, col24, col25, col26 = st.columns(5)
    
    col22.subheader('Mass Flow (kg/h)')
    result_Condenser = Condenser_prediction(df)
    series = pd.Series(result_Condenser[0])
    rounded_Condenser = round(series[0],2)
    col22.write(rounded_Condenser)

    col23.subheader('Vapour Fraction')
    result_VF = VF_prediction(df)
    series = pd.Series(result_VF[0])
    rounded_VF = round(series[0],4)
    col23.write(rounded_VF)

########################################################################
if __name__=='__main__':
    main()
    

with tab2:
    from PIL import Image
    #opening the image
    image = Image.open('HYSYS simulation.png')
    #displaying the image on streamlit app
    st.image(image, caption='Figure 1: Two-stage refrigeration cycle simulation on Aspen HYSYS.')
    image = Image.open('2StageVCR_Detailed.png')
    #displaying the image on streamlit app
    st.image(image, caption='Figure 2: Two-stage refrigeration system process flow diagram.')

mystyle = '''
    <style>
        p {
            text-align: justify;
        }
    </style>
    '''

st.write(mystyle, unsafe_allow_html=True)
    
with tab3:
    st.write('Computer simulation models have been the traditional use for remote monitoring and troubleshooting of chemical plants such that any variations in plant performance from the model predictions can be rapidly identified. The model can also simulate “what-if” scenarios, which include projecting what may hypothetically occur to the system under specific operating parameters in the future. The computerised model is typically a process simulation software and is often referred to as a “digital twin”, whereby it is an integrated, real-time digital version of the chemical plant. Process simulators are governed by first principles based physics framework that are used to detect crucial system behaviours such as absorption and compression. Once the performance of the simulation model is comparable to the actual plant\'s performance, it may be used to offer predictions for online decision support, such as suggesting the ideal operating conditions to troubleshooting and optimisation of the chemical plant.') 
    st.write('If the simulation models were to be employed for optimisation, the computational cost associated with process simulators would be much greater. This is because the optimisation algorithms, which are often recursive in nature, will repeatedly trigger on-demand simulation to arrive at a certain objective function. The high computational cost makes it infeasible for process simulators to be widely applied in optimisation related research. As a result, computation methods in process simulators are not particularly practical for field deployments in complex process plants that require quick decision making and online support. This is where data-driven models shine, as they can be computed up to a number of magnitudes more quickly than their first principle model counterpart.')
    st.write('The implementation of machine learning (ML) models, which are often referred to as black-box models are introduced to get around the low computational speed faced by traditional process simulators. Black-box models are constructed based on statistical information and produces an output that is solely based on the input data fed into it without any comprehension on the underlying principles. This method uses single or multivariate regression or interpolation to describe the first principle models using simplified mathematical equations.')
    st.write('In this work, a ML model was built to represent the behaviour of the two-stage refrigeration plant in Malaysia. This web app represents a hybrid model, which is often referred to as a gray-box model, whereby a physics-based first principle simulation model is combined with a data-driven machine learning (ML) model. A hybrid model not only relies on first principle domain knowledge to a certain extent, but it is also combined with statistical methods to infer trends and correlations that cannot be explained by first principles. Once data from the first principle model is obtained, it can be fed into the ML model for training. As a result, the ML model just consists of algebraic equations that can be computed quickly with ease. This allows the model to be deployed easily across a wider audience across organisations through web-based online dashboards, without the need for expensive process simulation software. This case utilises AspenTech\'s Aspen HYSYS V11 process simulator and Python for the ML model development using artificial neural networks and extreme gradient boosting.')
    
    
    
    
    
