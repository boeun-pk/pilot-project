
# 포트폴리오 파일럿 프로젝트 모음

이 저장소에는 다양한 YOLO 신경망 활용 파일럿 프로젝트가 포함되어 있습니다. 각 프로젝트는 실시간 객체 탐지, 추적, 분석 등을 목표로 합니다.

---

## 프로젝트 목록


### 1. 유동인구 카운트 YOLO 영상 파일럿 프로젝트
- **프로젝트 설명**: YOLO 모델을 사용하여 영상 속 유동인구를 실시간으로 감지하고, 입장 및 퇴장 수를 카운트하는 시스템.
- **주요 기능**: 사람 객체 탐지, 입장 및 퇴장 카운팅, 누적 인원 계산.
- **관련 링크**:
  - [YOLO 모델 다운로드](https://github.com/ultralytics/yolov5)
  - [OpenCV 공식 문서](https://docs.opencv.org/)

- **주피터 노트북 코드**
  - [구글 코렙](https://colab.research.google.com/drive/1gUCAYWWbg67DZaGdWa-_usnta1r9yziR#scrollTo=Kh0eiJKWA321)
- **시연 영상**:  
  - ![유동인구 카운트 영상](https://github.com/boeun-pk/project/blame/main/output3_with_counts.mp4)
  - ![image.png](https://github.com/boeun-pk/pilot-project/blob/main/yolo/count/image.png)

---

### 2. 엑스레이 사진에서 폐결절을 찾는 신경망
- **프로젝트 설명**:  x-ray 사진을 분류해서 폐 결절을 탐지하는 신경망 
- **주요 기능**: 폐결절 자동 탐지 
- **관련 링크**:
  - [구글 코렙](https://colab.research.google.com/drive/1xvnvfAQW9-nAEECK2f63nsFob7Zw9YcE?hl=ko)
  - [YOLO 모델 다운로드](https://github.com/ultralytics/yolov5)
- **시연 영상**:  
  - ![폐결절탐지](https://github.com/user-attachments/assets/e9c9ab96-5a19-4fc5-9d8b-a73f065f03d9)

--- 


### 3. 도둑감지 YOLO 신경망 활용 파일럿 프로젝트
- **프로젝트 설명**: 무인 감시 시스템을 위한 도둑 감지 기능. YOLO 모델을 통해 특정 위치에서의 불법 침입자를 실시간으로 탐지.
- **주요 기능**: 실시간 침입자 감지, 위험 알림, 비상 조치 기능.
- **관련 링크**:
  - [YOLO 모델 다운로드](https://github.com/ultralytics/yolov5)
  - [알림 시스템 API 문서](https://your-api-link.com/)
- **시연 영상**:  
  ![도둑 감지 영상](https://user-images.githubusercontent.com/yourusername/your-video-file2.mp4)

---

### 4. 차량 흐름 분석 파일럿 프로젝트
- **프로젝트 설명**: 도로 영상에서 차량 객체를 감지하고, 차량 흐름을 분석하여 교통 밀집도를 예측.
- **주요 기능**: 차량 객체 탐지, 밀집도 분석, 실시간 데이터 시각화.
- **관련 링크**:
  - [교통 데이터 API](https://traffic-api.com/)
- **시연 영상**:  
  ![차량 흐름 분석 영상](https://user-images.githubusercontent.com/yourusername/your-video-file3.mp4)

---

### 5. 얼굴 인식 출입 시스템 파일럿 프로젝트
- **프로젝트 설명**: YOLO와 얼굴 인식 모델을 이용한 출입 관리 시스템.
- **주요 기능**: 얼굴 인식을 통한 출입 허가, 기록 관리, 출입 로그.
- **관련 링크**:
  - [얼굴 인식 모델 다운로드](https://facerecognition-model.com/)
- **시연 영상**:  
  ![얼굴 인식 시스템 영상](https://user-images.githubusercontent.com/yourusername/your-video-file4.mp4)

---

### 6. 안전 헬멧 착용 감지 파일럿 프로젝트
- **프로젝트 설명**: 작업 현장에서 안전 헬멧 착용 여부를 감지하여 안전 규정을 준수하는지 확인.
- **주요 기능**: 헬멧 착용 감지, 안전 경고, 실시간 알림.
- **관련 링크**:
  - [헬멧 감지 데이터셋](https://helmet-dataset.com/)
- **시연 영상**:  
  ![헬멧 착용 감지 영상](https://user-images.githubusercontent.com/yourusername/your-video-file5.mp4)

---

## 설치 및 실행 방법

1. 필요한 라이브러리 설치:
   ```bash
   pip install opencv-python-headless ultralytics numpy
