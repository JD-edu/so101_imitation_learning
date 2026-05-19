import torch
import torch.nn as nn
import torch.optim as optim

# 1. 데이터 준비 (간단한 번역 쌍)
data = [
    ("나 가고 싶어", "i want to go"),
    ("그거 좋아", "i like it"),
    ("나 행복해", "i am happy"),
    ("너는 학생이야", "you are a student"),
    ("이거 뭐야", "what is this")
]

def build_vocab(sentences):
    # [1단계] 기본 특수 토큰 4개를 방 번호와 함께 미리 채워둡니다.
    # 번역기(Seq2Seq) 모델의 규칙을 위해 인위적으로 약속해둔 기호들입니다.
    vocab = {
        "<PAD>": 0,  # 문장 길이를 맞추기 위한 빈칸 채우기 기호
        "<SOS>": 1,  # Start Of Sequence: "이제 문장 시작한다!" 알림 기호
        "<EOS>": 2,  # End Of Sequence: "여기서 문장 끝났다!" 알림 기호
        "<UNK>": 3   # Unknown: 사전에 없는 낯선 단어 방어용 기호
    }
    # [2단계] 입력받은 모든 문장을 하나씩 꺼내어 반복합니다.
    for sent in sentences:   
        # [3단계] 하나의 문장을 띄어쓰기 기준으로 단어별로 쪼갭니다.
        # 예: "claim now" -> ["claim", "now"]
        words = sent.split()
        # [4단계] 쪼개진 단어들을 하나씩 순서대로 검사합니다.
        for word in words:
            # [5단계] (핵심) 만약 이 단어가 아직 사전에 등록되지 않았다면?
            if word not in vocab:
                # [6단계] 현재 사전의 총 길이(len(vocab))를 새 단어의 방 번호로 지정합니다.
                # 예: 처음에 특수 토큰 4개가 있으므로 len(vocab)은 4입니다. 
                # 따라서 첫 번째 새 단어의 방 번호는 자동으로 '4'가 됩니다.
                vocab[word] = len(vocab)
    # [7단계] 모든 문장 순회가 끝나면 완성된 단어 사전을 반환합니다.
    return vocab

# [1단계] 한국어 문장들만 따로 모아둘 빈 바구니(리스트)를 만듭니다.
korean_sentences = []

# [2단계] 전체 data 리스트에서 (한국어, 영어) 쌍을 하나씩 꺼내며 반복합니다.
for d in data:
    # d는 ("나 가고 싶어", "i want to go") 같은 튜플 형태입니다.
    # d[0]은 앞쪽의 '한국어 문장'이고, d[1]은 뒤쪽의 '영어 문장'이 됩니다.
    ko_sent = d[0] 
    #print(ko_sent)
    # [3단계] 골라낸 한국어 문장을 바구니에 차곡차곡 담습니다.
    korean_sentences.append(ko_sent)

# [확인] 여기까지 실행되면 korean_sentences는 아래와 같이 변합니다.
# ['나 가고 싶어', '그거 좋아', '나 행복해', '너는 학생이야', '이거 뭐야']
#print(korean_sentences)
# [4단계] 한국어 문장 목록만 담긴 바구니를 build_vocab 함수에 통째로 넘겨줍니다.
ko_vocab = build_vocab(korean_sentences)
print(ko_vocab)