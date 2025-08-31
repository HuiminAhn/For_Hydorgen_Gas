import os
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
import Data

# H2_save_path = 'model/H2_Rmodel.dat'
# CH4_save_path = 'model/CH4_Rmodel.dat'

def Test_and_Evaluate():
    # - 테스트 파일 불러오기

    taget_H2 = 'H2_converted_vol%'
    taget_CH4 = 'CH4_converted_vol%'

    # - 추론 데이터 목록 보여주기
    st.write(f'{Data.list_files('data/test')}')
        
    # - 추론 데이터 선택하기
    TeD_name = st.selectbox("추론하여 평가에 사용할 데이터를 선택해주세요.", Data.list_files('data/test'))
    st.write(f"선택한 추론데이터: {TeD_name}")
    
    if (TeD_name is None):
        st.write("디렉토리 내에 학습 데이터가 하나도 없습니다. 업로드 해주시길 바랍니다.")
    else:
        pred_button = st.button("추론과 평가 시작")
        
        if pred_button:
    
            test = pd.read_csv(f"data/test/{TeD_name}", index_col = 0)

            X_test = test.drop(labels = taget_H2, axis = 1)
            X_test = X_test.drop(labels = taget_CH4, axis = 1)

            Y_test_true_H2 = test[taget_H2]
            Y_test_true_CH4 = test[taget_CH4]

            # - 모델 초기화

            H2_Rmodel = XGBRegressor()
            CH4_Rmodel = XGBRegressor()

            # - 저장해둔 모델 불러오기

            H2_Rmodel.load_model('model/H2_Rmodel.dat')
            CH4_Rmodel.load_model('model/CH4_Rmodel.dat')

            # - 추론 수행

            test_H2_pred = H2_Rmodel.predict(X_test)
            test_CH4_pred = CH4_Rmodel.predict(X_test)

            # - 추론평가

            test_H2_mae = mean_absolute_error(Y_test_true_H2, test_H2_pred)
            test_H2_mape = mean_absolute_percentage_error(Y_test_true_H2, test_H2_pred)

            test_CH4_mae = mean_absolute_error(Y_test_true_CH4, test_CH4_pred)
            test_CH4_mape = mean_absolute_percentage_error(Y_test_true_CH4, test_CH4_pred)

            st.write(f'수소가스 mae : {test_H2_mae}, mape : {test_H2_mape}')
            st.write(f'메탄가스 mae : {test_CH4_mae}, mape : {test_CH4_mape}')
    
            test_H2_pred = pd.DataFrame(test_H2_pred)
            test_CH4_pred = pd.DataFrame(test_CH4_pred)
    
            test_H2_pred.columns = [f'{taget_H2}']
            test_CH4_pred.columns = [f'{taget_CH4}']
    
            # - 두 결과 컬럼 병합 및 저장
    
            test_result = pd.concat([test_H2_pred, test_CH4_pred], axis=1)
    
            test_result.to_csv(f'data/result/result_{TeD_name}', index = False)
            st.write("결론 데이터를 저장했습니다. 데이터 관리 메뉴에서 다운로드 받으실 수 있습니다.")
            return

# 0.13709452317312762 0.003067826577590255
# 0.13508969332301315 0.00429995357516302

def Test_To_Predict():
    # - 테스트 파일 불러오기

    taget_H2 = 'H2_converted_vol%'
    taget_CH4 = 'CH4_converted_vol%'

    # - 추론 데이터 목록 보여주기
    st.write(f'{Data.list_files('data/test')}')
        
    # - 추론 데이터 선택하기
    TeD_name = st.selectbox("실제로 예측할 데이터를 선택해주세요.", Data.list_files('data/test'))
    st.write(f"선택한 추론데이터: {TeD_name}")
    
    if (TeD_name is None):
        st.write("디렉토리 내에 학습 데이터가 하나도 없습니다. 업로드 해주시길 바랍니다.")
    else:
        pred_ana_button = st.button("예측과 분석 시작")
        
        if pred_ana_button:
    
            test = pd.read_csv(f"data/test/{TeD_name}", index_col = 0)

            X_test = test.drop(labels = taget_H2, axis = 1)
            X_test = X_test.drop(labels = taget_CH4, axis = 1)
    
            # - 모델 초기화

            H2_Rmodel = XGBRegressor()
            CH4_Rmodel = XGBRegressor()

            # - 저장해둔 모델 불러오기

            H2_Rmodel.load_model('model/H2_Rmodel.dat')
            CH4_Rmodel.load_model('model/CH4_Rmodel.dat')

            # - 예측 수행

            test_H2_pred = H2_Rmodel.predict(X_test)
            test_CH4_pred = CH4_Rmodel.predict(X_test)
    
            test_H2_pred = pd.DataFrame(test_H2_pred)
            test_CH4_pred = pd.DataFrame(test_CH4_pred)
    
            test_H2_pred.columns = [f'{taget_H2}']
            test_CH4_pred.columns = [f'{taget_CH4}']
    
            # - 두 결과 컬럼과 본래 예측용 데이터 병합 및 저장
    
            test_results = pd.concat([test_H2_pred, test_CH4_pred], axis = 1)
    
            #pd.read_csv(f"data/test/{TeD_name}", index_col = 0) 의 경우  index_col = 0 때문에 만약 concat에 쓰려면 .reset_index() 를 써서 인덱스 초기화를 해야 한다.
    
            X_test = X_test.reset_index()
            test_result = pd.concat([X_test, test_results], axis = 1)
    
            test_result.to_csv(f'data/result/Ana_result_{TeD_name}', index = False)
            st.write("결론 데이터를 저장했습니다. 데이터 관리 메뉴에서 다운로드 받으실 수 있습니다.")
    
            #H2 최대치, CH4 최저치 행 가져오기
            st.write("주어진 자료에서 수소가스를 최대치로 얻는 조건")
            st.dataframe(test_result[test_result['H2_converted_vol%'] == max(test_result['H2_converted_vol%'])])
    
            st.write("주어진 자료에서 차선택으로 메탄가스를 최저치로 얻는 조건")
            st.dataframe(test_result[test_result['CH4_converted_vol%'] == min(test_result['CH4_converted_vol%'])])
    
            return
    
def Set_Predict_Analysis_Expander():
    st.write('여기서는 모델을 통한 예측을 수행하고 그 예측을 통해 어느 조건에서 가장 품질 높은 수소를 얻을 수 있는지 알 수 있습니다.')
    st.write('참고로 모델의 예측은 혼동을 막기 위해 연속으로 진행할 수 없습니다.')
    
    st.write('테스트 파일로 추론 및 평가하기 는 어디까지나 평가에 속합니다.')
    st.write('테스트 파일로 예측 및 분석하기 는 실질적으로 자신이 원하는 조건의 테스트 데이터를 이용하여 결과 분석을 하는 방식입니다.')
    
    st.write('추론 및 평가하기의 결론 데이터는 "result_(선택한 테스트 파일 이름).csv" 라는 이름으로 저장됩니다.')
    st.write('예측 및 분석하기의 결론 데이터는 "Ana_result_(선택한 테스트 파일 이름).csv" 라는 이름으로 저장됩니다.')
    st.write('주의 : 만약 같은 이름의 테스트 파일로 추론 혹은 예측을 할 경우 같은 이름의 파일이 생겨 최신 파일이 자동적으로 덮어씌워집니다.')
    st.write('주의 : 결론 데이터는 절대적인 자료가 아니며 매 추론, 예측때마다 결과가 달라질 수 있습니다.')
    
    with st.expander('테스트 파일로 추론 및 평가하기'):
        Test_and_Evaluate()
    with st.expander('테스트 파일로 예측 및 분석하기'):
        Test_To_Predict()
