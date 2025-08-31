import os
import streamlit as st
import numpy as np
import pandas as pd

def list_files(directory):
    return os.listdir(directory)

def Upload_Train_Data():
    
    # - 학습 데이터 업로드하기
    uploaded_train_file = st.file_uploader("학습용 CSV 파일을 업로드하세요", type='csv')
    upload_name = st.text_input('업로드할 학습용 파일의 이름을 입력해 주시길 바랍니다. 입력 후 Enter를 누르면 저장됩니다.')

    if (uploaded_train_file is not None) and (upload_name is not None):
        data = pd.read_csv(uploaded_train_file)
        data.to_csv(f'data/train/{upload_name}.csv', index = False)
        st.write(f"학습 데이터 {upload_name}를 프로그램 내 디렉토리에 저장했습니다.")

def Delete_Train_Data():
    
    # - 학습 데이터 목록 보여주기
    st.write(f'{list_files('data/train')}')
        
    # - 학습 데이터 선택하기
    select_train_data = st.selectbox("삭제할 학습 데이터를 선택해주세요.", list_files('data/train'))
    st.write(f"삭제할 학습 데이터: {select_train_data}")
    
    if (select_train_data is None):
        st.write("디렉토리 내에 학습 데이터가 하나도 없습니다. 업로드 해주시길 바랍니다.")
    else:
        st.write("정말로 이 파일을 삭제하시겠습니까?")
        st.write("잘 생각해보시고 삭제 버튼을 눌러주시길 바랍니다.")
        
        delete_button = st.button("학습 데이터 삭제")
        
        if delete_button:
            os.remove(f'data/train/{select_train_data}')
            st.write(f"선택한 학습 데이터가 삭제되었습니다. {select_train_data}")

def Upload_Test_Data():
    
    # - 추론 데이터 업로드하기
    uploaded_test_file = st.file_uploader("추론용 CSV 파일을 업로드하세요", type='csv')
    upload_name = st.text_input('업로드할 추론용 파일의 이름을 입력해 주시길 바랍니다. 입력 후 Enter를 누르면 저장됩니다.')

    if (uploaded_test_file is not None) and (upload_name is not None):
        data = pd.read_csv(uploaded_test_file)
        data.to_csv(f'data/test/{upload_name}.csv', index = False)
        st.write(f"추론데이터 {upload_name}를 프로그램 내 디렉토리에 저장했습니다.")

def Delete_Test_Data():
    
    # - 추론 데이터 목록 보여주기
    st.write(f'{list_files('data/test')}')
    
    # - 추론 데이터 선택하기
    select_test_data = st.selectbox("삭제할 추론 데이터를 선택해주세요.", list_files('data/test'))
    st.write(f"삭제할 추론 데이터: {select_test_data}")
    
    if (select_test_data is None):
        st.write("디렉토리 내에 추론 데이터가 하나도 없습니다. 업로드 해주시길 바랍니다.")
    else:
        st.write("정말로 이 파일을 삭제하시겠습니까?")
        st.write("잘 생각해보시고 삭제 버튼을 눌러주시길 바랍니다.")
        
        delete_button = st.button("추론 데이터 삭제")
        
        if delete_button:
            os.remove(f'data/test/{select_test_data}')
            st.write(f"선택한 추론 데이터가 삭제되었습니다. {select_test_data}")
                
def Delete_Result_Data():
    
    # - 결론 데이터 목록 보여주기
    st.write(f'{list_files('data/result')}')
    
    # - 결론 데이터 선택하기
    select_result_data = st.selectbox("결론 데이터를 선택해주세요.", list_files('data/result'))
    st.write(f"선택한 결론 데이터: {select_result_data}")
    
    if (select_result_data is None):
        st.write("디렉토리 내에 결론 데이터가 하나도 없습니다. 업로드 해주시길 바랍니다.")
    else:
        st.write("정말로 이 파일을 삭제하시겠습니까?")
        st.write("잘 생각해보시고 삭제 버튼을 눌러주시길 바랍니다.")
        
        delete_button = st.button("삭제")
        
        if delete_button:
            os.remove(f'data/result/{select_result_data}')
            st.write(f"선택한 결론 데이터가 삭제되었습니다. {select_result_data}")

def Download_Result_Data():
    # - 결과 데이터 다운로드 받기
    
    st.write(f'{list_files('data/result')}')
    st.write("만약 'Ana_result_(선택한 테스트 파일 이름).csv'이라는 파일이 안 보이는 경우 새로고침을 해주시길 바랍니다.")
        
    # - 결론 데이터 선택하기
    select_result_data = st.selectbox("다운로드할 결론 데이터를 선택해주세요.", list_files('data/result'))
    st.write(f"선택한 결론 데이터: {select_result_data}")
    
    if (select_result_data is None):
        st.write("디렉토리 내에 결론 데이터가 하나도 없습니다. 추론 및 예측을 수행하면 데이터가 생성됩니다.")
    else:
        result_data = pd.read_csv(f"data/result/{select_result_data}", index_col = 0)
        st.dataframe(result_data)
        st.write("정말로 이 파일을 다운로드하시겠습니까?")
        st.write("잘 생각해보시고 다운로드 버튼을 눌러주시길 바랍니다.")
        
        download_result_button = st.download_button(label = "결과 파일 내려받기", data = result_data.to_csv(), file_name = f'{select_result_data}')
        
        if download_result_button:
            st.write(f"선택한 결론 데이터가 다운로드 되었습니다. {select_result_data}")
        
def Set_DataExpander():
    st.write('이 곳은 학습, 추론, 결론 데이터들을 관리하는 곳입니다.')

    with st.expander('학습 데이터 목록 보기'):
        st.write(f'{list_files('data/train')}')
    with st.expander('학습 데이터 업로드하기'):
        Upload_Train_Data()
    with st.expander('학습 데이터 삭제하기'):
        Delete_Train_Data()

    with st.expander('추론 데이터 목록 보기'):
        st.write(f'{list_files('data/test')}')
    with st.expander('추론 데이터 업로드하기'):
        Upload_Test_Data()
    with st.expander('추론 데이터 삭제하기'):
        Delete_Test_Data()
        
    with st.expander('결론 데이터 목록 보기'):
        st.write(f'{list_files('data/result')}')
    with st.expander('결론 데이터 삭제하기'):
        Delete_Result_Data()
    with st.expander('결론 데이터 다운로드 하기'):
        Download_Result_Data()