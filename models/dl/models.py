import torch
import torch.nn as nn



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

    def forward(self, word_embed):
        




class Classifier(nn.Module):

    def __init__(self, config,vocab):
        super(Classifier, self).__init__()

        self.embed = nn.Embedding(len(vocab), config.d_embed)
        model.embed.weight.data.copy_(TEXT.vocab.vectors)
        self.encoder = RNNEncoder(config)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        seq_in_size = 2*config.d_hidden
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
            Linear(seq_in_size, 1),
            self.sigmoid)

    def forward(self, batch):
        question_embed = self.embed(batch.question_text)
        question_hid = self.encoder(question_embed)
        score = self.out(question_hid)
        return score
