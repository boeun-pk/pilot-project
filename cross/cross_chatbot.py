import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 포트폴리오 관련 질문과 답변 데이터
questions = [
    "포트폴리오 주제가 뭔가요?",
    "모델은 어떤 걸 썼나요?",
    "프로젝트 인원은 몇명인가요?",
    "프로젝트 기간은 어떻게 되나요?",
    "조장이 누구인가요?",
    "데이터는 어디서 구했나요?",
    "무슨 데이터를 사용했나요?",
    "프로젝트 하는데 어려움은 없었나요?"
]

answers = [
    "yolo 모델을 활용해서 통행 약자분들이 통행할 때 도움을 줄 수 있는 시스템을 만드는 것입니다. ",
    "yolo 모델 8 버전을 사용했습니다.",
    "이동근, 박보은, 석승연, 김연우, 손주용으로 총 5명입니다.",
    "24년 10월 28일 ~ 11월 18일로 총 3주입니다.",
    "조장은 손주용입니다.",
    "AI 허브에서 구해서 사용했습니다.",
    "보행자 시점의 인도 보행 이미지 데이터를 사용했습니다.",
    "어려웠습니다."
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})



# Streamlit 인터페이스
st.title("통행 약자 인도 보행 안전 어시스턴트 서비스")

# 이미지 표시
st.image("cross_image.png", caption="Welcome to the Restaurant Chatbot", use_column_width=True)

st.write("프로젝트에 대한 질문을 입력해주세요 예: 포트폴리오 주제가 뭔가요? ")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")
