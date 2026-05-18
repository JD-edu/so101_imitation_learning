import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 저장된 모델 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained("./kogpt2_chatbot")
model = GPT2LMHeadModel.from_pretrained("./kogpt2_chatbot").to(device)

def chat():
    model.eval()
    print("챗봇과 대화를 시작합니다. '종료'를 입력하면 끝납니다.")
    
    with torch.no_grad():
        while True:
            q = input("나: ").strip()
            if q == "종료": break
            
            # 질문 포맷팅 (<usr>질문<unused1><sys>)
            input_ids = tokenizer.encode("<usr>" + q + "<unused1>" + "<sys>", return_tensors='pt').to(device)
            
            # 답변 생성
            output = model.generate(
                input_ids,
                max_length=50,
                repetition_penalty=2.0, # 반복 방지
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                use_cache=True
            )
            
            # 생성된 텍스트 중 답변 부분만 추출
            res = tokenizer.decode(output[0])
            # <sys> 이후의 텍스트만 가져오고 </s>는 제거
            response = res.split("<sys>")[-1].replace("</s>", "").strip()
            print(f"챗봇: {response}")

if __name__ == "__main__":
    chat()