import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import time

# 전체 레이아웃을 넓게 설정
st.set_page_config(layout="wide")

# 제목 설정
st.title("비디오 사물 검출 앱")

# 모델 파일 업로드
model_file = st.file_uploader("모델 파일을 업로드하세요", type=["pt"])
if model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model_file:
        temp_model_file.write(model_file.read())
        model_path = temp_model_file.name
    model = YOLO(model_path)
    st.success("모델이 성공적으로 로드되었습니다.")

# 비디오 파일 업로드
uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])

# 원본 비디오와 결과 비디오 레이아웃
col1, col2 = st.columns(2)

with col1:
    st.header("원본 영상")
    if uploaded_file:
        st.video(uploaded_file)
    else:
        st.write("원본 영상을 표시하려면 비디오 파일을 업로드하세요.")

# 버튼 스타일 설정
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4d4d4d;
        color: #ffffff;
        font-weight: bold;
        padding: 12px 24px;
        font-size: 16px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 실시간 프레임 표시
if st.button("사물 검출 실행") and uploaded_file and model_file:
    # 파일 읽기 설정
    with tempfile.NamedTemporaryFile(delete=False) as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    cap = cv2.VideoCapture(temp_input_path)
    result_placeholder = col2.empty()  # 검출 결과를 표시할 공간

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델로 예측 수행
        results = model(frame)
        detections = results[0].boxes if len(results) > 0 else []

        # 검출된 객체 표시
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 실시간 프레임을 Streamlit에 표시
        result_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 약간의 지연을 추가하여 비디오 속도 제어
        # 비디오 속도 제어 - 값을 작게 설정하여 더 빠르게 재생
        time.sleep(0.00001)


    cap.release()
    st.success("사물 검출이 완료되었습니다.")
