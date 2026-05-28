import pandas as pd
import re 

class ChatbotVocab:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.vocab_size = 4

    def tokenize(self, text):
        text_re = re.sub(r"[^가-힣a-zA-Z0-9]", " ", text)
        #print(text_re)
        text_split = text_re.split()
        #print(text_split)
        return text_split
    
    # 단어사전 vocab을 만들기 
    def build_vocab(self, sentences):
        for sentence in sentences:
            #print(sentence)
            for word in self.tokenize(sentence):
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size # 딕셔너리 단어:인덱스번호
                    self.idx2word[self.vocab_size] = word # 딕셔너리 인덱스번호:단어
                    self.vocab_size += 1 
        

df = pd.read_csv('ChatbotData.csv')
#print(df)

vocab = ChatbotVocab()
#vocab.tokenize("I am student. %$%^. 뭐라고 하는 거야@##@$#")

q_list = df['Q'].tolist()
#print(q_list)
a_list = df['A'].tolist()
#print(a_list)

vocab.build_vocab(q_list+a_list)
