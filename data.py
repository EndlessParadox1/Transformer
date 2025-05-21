from torchtext.datasets import IWSLT2017
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = IWSLT2017(split='train', language_pair=('en', 'zh'))
valid_iter = IWSLT2017(split='valid', language_pair=('en', 'zh'))

tokenizer_en = get_tokenizer("spacy", language="en_core_web_sm")
tokenizer_zh = get_tokenizer("spacy", language="zh_core_web_sm")

# 构建词表
def yield_tokens(data_iter, tokenizer, index):
    for data in data_iter:
        yield tokenizer(data[index])

vocab_en = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer_en, 0), specials=["<pad>", "<unk>", "<sos>", "<eos>"])
vocab_zh = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer_zh, 1), specials=["<pad>", "<unk>", "<sos>", "<eos>"])

vocab_en.set_default_index(vocab_en["<unk>"])
vocab_zh.set_default_index(vocab_zh["<unk>"])
