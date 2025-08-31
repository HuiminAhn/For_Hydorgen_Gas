import os
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
import Data

def Model_Training():
    
    # - 데이터 전처리
    
    taget_H2 = 'H2_converted_vol%'
    taget_CH4 = 'CH4_converted_vol%'

    # - 학습 데이터 목록 보여주기
    st.write(f'{Data.list_files('data/train')}')
    
    st.write("버튼을 누르면 모델이 학습을 시작하고 모델 파일을 지정된 디렉토리에 저장합니다.")
    st.write("만약 이미 디렉토리에 모델이 존재하는 경우 새로고침 됩니다.")
    st.write("주의 : 학습 과정 중 종료하면 데이터의 손실이 있을 수 있습니다.")
        
    # - 학습 데이터 선택하기
    TrD_name = st.selectbox("학습 데이터를 선택해주세요.", Data.list_files('data/train'))
    st.write(f"선택한 학습 데이터: {TrD_name}")
        
    if (TrD_name is None):
        st.write("디렉토리 내에 학습 데이터가 하나도 없습니다. 업로드 해주시길 바랍니다.")
    else:
        train_button = st.button("모델 학습")
        
        if train_button:
            
            data = pd.read_csv(f"data/train/{TrD_name}", index_col = 0)

            train, val = train_test_split(data, train_size = 0.8, shuffle = True)

            train_H2 = train.drop(labels = taget_CH4, axis = 1)
            X_train_H2 = train_H2.drop(labels = taget_H2, axis = 1)
            Y_train_H2 = train_H2[taget_H2]

            train_CH4 = train.drop(labels = taget_H2, axis = 1)
            X_train_CH4 = train_CH4.drop(labels = taget_CH4, axis = 1)
            Y_train_CH4 = train_CH4[taget_CH4]

            true_H2 = val[taget_H2]
            true_CH4 = val[taget_CH4]

            val = val.drop(taget_H2, axis = 1)  
            val = val.drop(taget_CH4, axis = 1)
    
            st.write("학습 데이터와 모델 검증용 데이터를 분리했습니다.")
    
# - 모델 제조 및 저장

# param_grid={
#     'learning_rate':[0.01, 0.015], 
#     'max_depth':[0, 10], 
#     'n_estimators':[1000, 1500]
#     }

            xgb_params = {
                "n_estimators": 1000,
                "learning_rate": 0.01,
                "max_depth": 10,
                "max_leaves": 100,
                "min_child_weight": 10,
                "subsample": 0.5,
                "colsample_bytree": 0.6,
                "colsample_bylevel": 0.6,
                "gamma": 18,
                "eval_metric": "mape",
                "early_stopping_rounds": 50,
                "random_state": 514
            }

            H2_Rmodel = XGBRegressor(**xgb_params)

            H2_eval_set = [(val, true_H2)]

            CH4_Rmodel = XGBRegressor(**xgb_params)

            CH4_eval_set = [(val, true_CH4)]
    
            st.write("예측 모델을 생성했습니다.")

# best parameters :  {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1000}
# best score :  -0.0311
# best parameters :  {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1000}
# best score :  -0.0413

# - 모델 학습 

            H2_Rmodel.fit(X = X_train_H2,
                    y = Y_train_H2,
                    eval_set=H2_eval_set,
                    verbose=True)

            CH4_Rmodel.fit(X = X_train_CH4,
                    y = Y_train_CH4,
                    eval_set=CH4_eval_set,
                    verbose=True)
    
            st.write("모델 학습을 완료했습니다.")

            H2_pred = H2_Rmodel.predict(val)
            CH4_pred = CH4_Rmodel.predict(val)

            H2_mae = mean_absolute_error(true_H2, H2_pred)
            H2_mape = mean_absolute_percentage_error(true_H2, H2_pred)

            CH4_mae = mean_absolute_error(true_CH4, CH4_pred)
            CH4_mape = mean_absolute_percentage_error(true_CH4, CH4_pred)

# 1.4415602259347589 0.031151609973779488
# 1.2765604896885991 0.041467474649272626

            st.write(f'수소가스 mae : {H2_mae}, mape : {H2_mape}')
            st.write(f'메탄가스 mae : {CH4_mae}, mape : {CH4_mape}')
    
    # - 모델 갱신
            
            H2_Rmodel.save_model('model/H2_Rmodel.dat')
            CH4_Rmodel.save_model('model/CH4_Rmodel.dat')

            st.write("지정한 디렉토리에 학습된 모델들을 저장하였습니다.")
            return

def Set_Model_Training_Expander():
    st.write('여기서는 학습데이터를 통한 모델의 학습 및 평가를 합니다. 프로그램 내 모델은 단 하나만 존재할 수 있습니다.')
    
    st.write('이 과정은 학습할 데이터를 선택하고 모델을 자동적으로 만들어 저장합니다.')
    st.write('주의 : 학습된 모델 만들기를 누르면 모델 제작 완료 및 갱신 혹은 프로그램 종료가 될 때까지 주어진 과정을 멈추지 않습니다.')
    st.write('모델의 주 성능평가 지표는 mae, mape입니다. 확인하시고 모델 선정을 해주시길 바랍니다.')
    
    with st.expander('학습한 모델 만들기'):
        Model_Training()