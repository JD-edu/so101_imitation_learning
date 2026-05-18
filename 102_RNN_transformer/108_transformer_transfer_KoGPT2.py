import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# 1. 고정된 설정값
Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
PAD = '<pad>'
SENT = '<unused1>'

# 2. 토크나이저 및 모델 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

# 3. 데이터셋 클래스
class ChatDataset(Dataset):
    def __init__(self, chats, max_len=40):
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.bos = BOS
        self.eos = EOS
        self.tokenizer = tokenizer

    def __len__(self): return len(self._data)

    def __getitem__(self, idx):
        record = self._data.iloc[idx]
        q = record['Q']
        a = record['A']
        
        # 질문 + 답변 형태의 시퀀스 생성
        q_tkn = self.tokenizer.tokenize(self.q_token + q + SENT)
        a_tkn = self.tokenizer.tokenize(self.a_token + a + self.eos)
        
        # 최대 길이에 맞춰 자르기 및 패딩
        tokens = q_tkn + a_tkn
        mask = [0] * len(q_tkn) + [1] * len(a_tkn) # 답변 부분만 학습하도록 마스킹
        
        labels = [self.tokenizer.pad_token_id] * len(q_tkn) + self.tokenizer.convert_tokens_to_ids(a_tkn)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # 패딩 처리
        padding_len = self.max_len - len(token_ids)
        token_ids += [self.tokenizer.pad_token_id] * padding_len
        labels += [self.tokenizer.pad_token_id] * padding_len
        
        return torch.tensor(token_ids), torch.tensor(labels)

# 4. 학습 실행
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
df = pd.read_csv('ChatbotData.csv')
dataset = ChatDataset(df)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
model.train()

for epoch in range(5): # 사전학습 모델이므로 5~10회면 충분합니다.
    for batch in loader:
        input_ids, labels = batch
        input_ids, labels = input_ids.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# 모델 및 토크나이저 저장
model.save_pretrained("./kogpt2_chatbot")
tokenizer.save_pretrained("./kogpt2_chatbot")