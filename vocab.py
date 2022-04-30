import os
import json
import nltk
from transformers import BertTokenizer


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        assert len(self.word2idx) == len(self.idx2word)
        return len(self.word2idx)


def load_vocab(path):
    with open(path) as f:
        d = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = d['word2idx']
    vocab.idx2word = d['idx2word']
    vocab.idx = d['idx']
    return vocab


def get_vocab(txt_enc_type, vocab_path, data_name, bert_type=None):
    if txt_enc_type == 'rnn':
        vocab = load_vocab(os.path.join(vocab_path,
                                        f'{data_name}_precomp_vocab.json'))
        vocab.add_word('<mask>')
        tokenizer = nltk.tokenize.word_tokenize
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_type)
        vocab = tokenizer.vocab
    return vocab, tokenizer