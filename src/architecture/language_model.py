import random

from torch import nn
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from architecture.utils import freeze_model, unfreeze_model

nlp = English()


# TODO check where is language model CPU or GPU
class LanguageModel(nn.Module):
    def __init__(self, type, embedding_size, linear1_out=256, out_features=100,
                 vocab=None, vocab_size=500, device='cuda' if torch.cuda.is_available() else 'cpu',
                 pretrained_model=None, freq_update=1):
        """

        :param type: string 'linear' or 'lstm'
        :param embedding_size:
        :param linear1_out:
        :param vocab:
        :param vocab_size:
        """
        super().__init__()
        self.frozen = False
        self.freq_update = freq_update
        self.out_features = out_features
        self.word_to_ix = dict()
        self.type = type
        self.tokenizer = Tokenizer(nlp.vocab)
        self.embedding_size = embedding_size

        if vocab:
            self.word_to_ix = vocab.stoi
            self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_size,
                                                scale_grad_by_freq=False)
            self.embedding_layer.weight.data.copy_(vocab.vectors)
            self.embedding_layer.require_grad = False
        else:
            self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size,
                                                scale_grad_by_freq=False)

        if self.type == 'linear':
            self.type = 'linear'
            self.linear1 = nn.Linear(in_features=embedding_size, out_features=linear1_out)
            self.linear2 = nn.Linear(in_features=linear1_out, out_features=out_features)
        elif self.type == 'lstm':
            self.type = 'lstm'
            self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=out_features, bias=False)
        else:
            raise NotImplementedError

        self.device = device
        self.to(device)

        if pretrained_model:
            self.load_state_dict(torch.load(pretrained_model))
            self.eval()
            self.freeze()

    def add_tokens(self, token):
        raise KeyError(token)
        idx = len(self.word_to_ix)
        if idx in self.word_to_ix.values():
            idx = min(set(range(100000)).difference(self.word_to_ix.values()))
        self.word_to_ix.update({token: idx})

    def prepare_sentence(self, sentence):
        if isinstance(sentence, str):
            tokens = self.tokenizer(sentence.lower())
            try:
                return torch.LongTensor([self.word_to_ix[t.text] for t in tokens])
            except KeyError as new_t:
                self.add_tokens(new_t.args[0])
                return self.prepare_sentence(sentence)
        elif isinstance(sentence, list):
            batch_sentence = [self.prepare_sentence(s) for s in sentence]
            return batch_sentence
        else:
            raise TypeError('Sentence should be passed as a string or list of string')

    # TODO Compute forward pass in one pass : need to handle pad token
    def forward(self, sentence):
        if isinstance(sentence, str):
            sentence = [sentence]

        s = self.prepare_sentence(sentence)
        s = rnn_utils.pad_sequence(s)
        s = self.embedding_layer(s.to(self.device))
        if self.type == 'linear':
            s = self.linear1(s)
            s = F.relu(s)
            s = self.linear2(s)
            s = F.relu(s)
            out = s.mean(dim=1, keepdim=True)
        elif self.type == 'lstm':
            output, (h, c) = self.lstm(s)
            out = torch.tanh(h).squeeze(0)
        else:
            raise NotImplementedError('Language policy_network type should be linear or LSTM')
        return out

    def freeze_sometimes(self, episode):
        """
        Unfreeze model for update one episode every self.freq_update
        Parameters
        ----------
        episode : episode number

        Returns None
        -------

        """
        if episode % self.freq_update == 0:
            self.unfreeze()
        elif episode % self.freq_update == 1:
            self.freeze()

    def freeze(self):
        freeze_model(self)
        self.frozen = True

    def unfreeze(self):
        unfreeze_model(self)
        self.frozen = False


if __name__ == '__main__':
    sentence = ['Hi I am a people', 'Hi, How are you ?']
    l1 = LanguageModel(type='linear', embedding_size=50)

    l2 = LanguageModel(type='lstm', embedding_size=50)

    t1 = l1(sentence)

    t2 = l2(sentence)

    assert t1.size() == t2.size()
#
# import random
#
# from torch import nn
# import torch
# import torch.nn.functional as F
# import torch.nn.utils.rnn as rnn_utils
#
# from spacy.tokenizer import Tokenizer
# from spacy.lang.en import English
#
# from architecture.utils import freeze_model, unfreeze_model
#
# nlp = English()
#
#
# # TODO check where is language model CPU or GPU
# class LanguageModel(nn.Module):
#     def __init__(self, type, embedding_size, linear1_out=256, out_features=100,
#                  vocab=None, vocab_size=500, device='cuda' if torch.cuda.is_available() else 'cpu',
#                  pretrained_model=None, freq_update=1):
#         """
#
#         :param type: string 'linear' or 'lstm'
#         :param embedding_size:
#         :param linear1_out:
#         :param vocab:
#         :param vocab_size:
#         """
#         super().__init__()
#         self.frozen = False
#         self.freq_update = freq_update
#         self.out_features = out_features
#         self.word_to_ix = dict()
#         self.type = type
#         self.tokenizer = Tokenizer(nlp.vocab)
#         self.embedding_size = embedding_size
#
#         if vocab:
#             self.word_to_ix = vocab.stoi
#             self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_size,
#                                                 scale_grad_by_freq=False)
#             self.embedding_layer.weight.data.copy_(vocab.vectors)
#             self.embedding_layer.require_grad = False
#         else:
#             self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size,
#                                                 scale_grad_by_freq=False)
#
#         if self.type == 'linear':
#             self.type = 'linear'
#             self.linear1 = nn.Linear(in_features=embedding_size, out_features=linear1_out)
#             self.linear2 = nn.Linear(in_features=linear1_out, out_features=out_features)
#         elif self.type == 'lstm':
#             self.type = 'lstm'
#             self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=out_features, bias=False)
#         else:
#             raise NotImplementedError
#
#         self.device = device
#         self.to(device)
#
#         if pretrained_model:
#             self.load_state_dict(torch.load(pretrained_model))
#             self.eval()
#             self.freeze()
#
#     def add_tokens(self, token):
#         idx = len(self.word_to_ix)
#         if idx in self.word_to_ix.values():
#             idx = min(set(range(100000)).difference(self.word_to_ix.values()))
#         self.word_to_ix.update({token: idx})
#
#     def prepare_sentence(self, sentence):
#         if isinstance(sentence, str):
#             tokens = self.tokenizer(sentence.lower())
#             try:
#                 return torch.LongTensor([self.word_to_ix[t.text] for t in tokens]), len(tokens)
#             except KeyError as new_t:
#                 self.add_tokens(new_t.args[0])
#                 return self.prepare_sentence(sentence)
#         elif isinstance(sentence, list):
#             batch_sentence = [self.prepare_sentence(s) for s in sentence]
#             return batch_sentence
#         else:
#             raise TypeError('Sentence should be passed as a string or list of string')
#
#     # TODO Compute forward pass in one pass : need to handle pad token
#     def forward(self, sentence):
#         if isinstance(sentence, str):
#             sentence = [sentence]
#
#         sentence_vectors, seq_size = zip(*self.prepare_sentence(sentence))
#         s = rnn_utils.pad_sequence(sentence_vectors)
#         s = self.embedding_layer(s.to(self.device))
#         if self.type == 'linear':
#             s = self.linear1(s)
#             s = F.relu(s)
#             s = self.linear2(s)
#             s = F.relu(s)
#             out = s.mean(dim=1, keepdim=True)
#         elif self.type == 'lstm':
#             output, _ = self.lstm(s)
#             idx = (torch.tensor(seq_size) - 1).to(self.device)
#             idx = idx.unsqueeze(1).repeat(1, 50).unsqueeze(1)
#             out = output.gather(dim=0, index=idx)
#             out = torch.tanh(out).squeeze()
#         else:
#             raise NotImplementedError('Language policy_network type should be linear or LSTM')
#         return out
#
#     def freeze_sometimes(self, episode):
#         """
#         Unfreeze model for update one episode every self.freq_update
#         Parameters
#         ----------
#         episode : episode number
#
#         Returns None
#         -------
#
#         """
#         if episode % self.freq_update == 0:
#             self.unfreeze()
#         elif episode % self.freq_update == 1:
#             self.freeze()
#
#     def freeze(self):
#         freeze_model(self)
#         self.frozen = True
#
#     def unfreeze(self):
#         unfreeze_model(self)
#         self.frozen = False
#
#
# if __name__ == '__main__':
#     sentence = ['Hi I am a people', 'Hi, How are you ?']
#     l1 = LanguageModel(type='linear', embedding_size=50)
#
#     l2 = LanguageModel(type='lstm', embedding_size=50)
#
#     t1 = l1(sentence)
#
#     t2 = l2(sentence)
#
#     assert t1.size() == t2.size()
