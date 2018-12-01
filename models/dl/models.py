import torch
import torch.nn as nn
from torch.nn.utils import rnn
from torch.autograd import Variable

class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class RNNEncoder(nn.Module):
    def __init__(self, config):
        super(RNNEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size = config.d_embed,
                            hidden_size = config.d_hidden,
                            batch_first=True)

    def forward(self, question_embed,x_length):
        x_emb_p = rnn.pack_padded_sequence(question_embed, x_length, batch_first=True)
        out_pack, (ht, ct) = self.rnn(x_emb_p)
        # out = rnn.pad_packed_sequence(out_pack, batch_first=True)
        return ht[-1]







class Classifier(nn.Module):

    def __init__(self, config,vocab):
        super(Classifier, self).__init__()
        # self.device = torch.device('cuda:{}'.format(config.gpu))
        self.embed = nn.Embedding(len(vocab), config.d_embed)
        self.embed.weight.data.copy_(vocab.vectors)
        self.encoder = RNNEncoder(config)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        seq_in_size = config.d_hidden
        lin_config = [seq_in_size]*2
        self.out = nn.Sequential(
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(seq_in_size, 2),
            self.sigmoid)

    def forward(self, batch):
        (x, x_length) = batch.question_text
        question_embed = self.embed(x)
        question_hid = self.encoder(question_embed, x_length)
        score = self.out(question_hid)
        return score
