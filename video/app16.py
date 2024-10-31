import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np
from io import BytesIO

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

# 전체 레이아웃을 컨테이너로 감싸기
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.header("원본 영상")
        if uploaded_file is not None:
            st.video(uploaded_file)
        else:
            st.write("원본 영상을 표시하려면 비디오 파일을 업로드하세요.")

    with col2:
        st.header("사물 검출 결과 영상")
        result_placeholder = st.empty()

        # 검출 결과가 표시될 빈 화면
        result_placeholder.markdown(
            """
            <div style='width:100%; height:620px; background-color:#d3d3d3; display:flex; align-items:center; justify-content:center; border-radius:5px;'>
                <p style='color:#888;'>여기에 사물 검출 결과가 표시됩니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# 사물 검출 버튼 클릭 이벤트 처리
if st.button("사물 검출 실행") and uploaded_file and model_file:
    # 비디오 처리
    cap = cv2.VideoCapture(uploaded_file.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 메모리 내에서 비디오를 처리하고 저장
    output_video = BytesIO()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델로 예측 수행
        results = model(frame)
        detections = results[0].boxes if len(results) > 0 else []

        # 검출 결과에 대한 바운딩 박스와 라벨 추가
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 메모리 내에서 비디오 프레임 작성
        out.write(frame)

    cap.release()
    out.release()

    # 결과 비디오를 스트리밍 가능한 형태로 변환하여 스트림릿에 표시
    output_video.seek(0)
    result_placeholder.video(output_video)
    st.success("사물 검출이 완료되어 오른쪽에 표시됩니다.")
