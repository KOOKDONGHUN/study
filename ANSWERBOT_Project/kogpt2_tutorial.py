import torch
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer

tok_path = get_tokenizer()
print(tok_path) # C:\Users\bitcamp/kogpt2/kogpt2_news_wiki_ko_cased_818bfa919d.spiece

model, vocab = get_pytorch_kogpt2_model()
print(model)
print(vocab) # Vocab(size=50000, unk="<unk>", reserved="['<pad>', '<s>', '</s>']")

tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)

sentence = '2019년 한해를 보내며,'

toked = tok(sentence)
# print(vocab.keys()) # AttributeError: 'BERTVocab' object has no attribute 'keys'
print(type(vocab)) # <class 'gluonnlp.vocab.bert.BERTVocab'>
print(vocab.bos_token) # <s>

while !(len(toked)>=15):
    input_ids = torch.tensor([vocab.bos_token],)