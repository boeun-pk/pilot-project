import streamlit as st

# 전체 레이아웃을 넓게 설정
st.set_page_config(layout="wide") 

# 제목 설정
st.title("비디오 사물 검출 앱")

# 파일 업로드 
uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])


# 전체 레이아웃을 컨테이너로 감싸기
with st.container():   # with 절로 하나의 기능을 하는 코드를 묶어주는 것(가독성 높이기) - sql with절과 다름 
    col1, col2 = st.columns(2)  # 두개의 열을 균등하게 분배하여 넓게 표시

    with col1:
        st.header("원본 영상")   # col1 영역의 제목 
        if uploaded_file is not None:    # 영상이 업로드되었다면 
            st.video(uploaded_file)   # 영상을 플레이해라 
        else:
            st.write("원본 영상을 표시하려면 비디오 파일을 업로드하세요.") # 안됐으면 출력 

    with col2:
        st.header("사물 검출 결과 영상")   # col2 영역의 제목 
        if "processed_video" in st.session_state: # session_state 안에 processed_video 라는 키가 존재하면 
                                                  # = 사물 검출 완료된 비디오가 있다면 
            st.video(st.session_state["processed_video"])  # 해당 비디오 플레이
        else:
            st.write("여기에 사물 검출 결과가 표시됩니다.")

# 사물 검출 버튼 추가
if st.button("사물 검출 실행"): # 버튼 누르면 
    if uploaded_file is not None:  # 업로드된 파일이 있다면 
        st.session_state["processed_video"] = uploaded_file  # 검출된 영상을 사용 
        st.success("사물 검출이 완료되어 오른쪽에 표시됩니다.")  # 메시지 출력 
    else:
        st.warning("사물 검출을 실행하려면 비디오 파일을 업로드하세요.")