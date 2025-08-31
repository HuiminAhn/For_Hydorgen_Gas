import os
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
import subprocess
import sys
import Data
import Predict_Evaluate
import Training_model
from PIL import Image

def Setting_Main():
    st.title("수소가스 품질 예측 모델 프로그램")
    st.write("이 프로그램은 규격에 맞는 데이터를 통해 모델을 학습, 추론하여 성능최적화를 시킨 후")
    st.write("사용자가 생각한 조건 들 중 어떤 상황이 수소가스 최대 수득률을 예측 할 수 있는지")
    st.write("아니면 그나마 메탄가스 최저 발생률을 예측 할 수 있는지 알아보는 프로그램입니다.")

    st.write("csv 파일에서 참고 할 수 있는 규격은 아래와 같으며 반드시 해당 규격을 가진 csv 확장자 파일을 사용하셔야 합니다.")
    sample_image = Image.open('csv_example.jpg')
    st.image(sample_image)
    
    st.write("데이터 관리, 모델 학습, 예측 및 분석을 하시려면 아래 확장되는 바를 눌러주시길 바랍니다.")
    
    with st.expander("데이터 관리"):
        Data.Set_DataExpander()
    with st.expander("모델 학습"):
        Training_model.Set_Model_Training_Expander()
    with st.expander("예측 및 분석"):
        Predict_Evaluate.Set_Predict_Analysis_Expander()

Setting_Main()