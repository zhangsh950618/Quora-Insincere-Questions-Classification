import os
import time
import glob

import torch
import torch.optim as O
import torch.nn as nn

from torchtext import data
from torchtext import datasets

from util import get_args,SplitReversibleField
from models import Classifier
import logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s   %(levelname)s   %(message)s')
args = get_args()
torch.cuda.set_device(args.gpu)
device = torch.device('cuda:{}'.format(args.gpu))

TEXT = SplitReversibleField(sequential=True, tokenize='spacy',lower=args.lower, include_lengths=True,use_vocab=True)
LABEL = data.Field(sequential=False,use_vocab=False,is_target=True)


# ------------------------  buding datasets ------------------------
logging.info("Building datasets, loding word vectors")
train, dev = data.TabularDataset.splits(
        skip_header=True,
        path=args.data_path,
        train='sample.csv',
        validation='sample.csv',
        format='csv',
        fields=[('qid',None),('question_text', TEXT), ('label', LABEL)])

train_iter, dev_iter = data.Iterator.splits(
        (train, val), sort_key=lambda x: len(x.question_text),
        batch_sizes=(32, 32),
        sort_within_batch=True, repeat=False)


# ------------------------  buding vocab ------------------------
logging.info("loading vocab")
TEXT.build_vocab(train, vectors="glove.6B.100d")

# ------------------------  buding network ------------------------
logging.info("building network")
model = Classifier(args)
model.embed.weight.data.copy_(TEXT.vocab.vectors)
model.to(device)
criterion = nn.CrossEntropyLoss()
opt = O.Adam(model.parameters(), lr=args.lr)

# ------------------------  start training ------------------------
logging.info("start training")

header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))

print(header)
iterations = 0
start = time.time()

for epoch in range(args.epochs):
    n_correct, n_total = 0, 0
    train_iter.init_epoch()
    for batch_idx, batch in enumerate(train_iter):
        model.train(); opt.zero_grad()
        iterations += 1
        predict = model(batch)
        loss = criterion(answer, batch.label)
        loss.backward(); opt.step()

        if iterations % args.dev_every == 0:
            model.eval()
            dev_iter.init_epoch()
            n_dev_correct, dev_loss = 0, 0
            with torch.no_grad():
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                     answer = model(dev_batch)
                     n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()) == dev_batch.label).sum().item()
                     dev_loss = criterion(answer, dev_batch.label)
            dev_acc = 100. * n_dev_correct / len(dev)
            print(dev_log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.item(), dev_loss.item(), train_acc, dev_acc))
        elif iterations % args.log_every == 0:
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.item(), ' '*8, n_correct/n_total*100, ' '*12))
