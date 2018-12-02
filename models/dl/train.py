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
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s   %(levelname)s   %(message)s')
args = get_args()
torch.cuda.set_device(args.gpu)
device = torch.device('cuda:{}'.format(args.gpu))

# ------------------------  split datasets ------------------------
all = pd.read_csv(os.path.join(args.data_path, "all.csv"))

train_x,dev_x,_,_ = train_test_split(all.values.tolist(),[0] * len(all),test_size=0.2,random_state=0)

train_x = pd.DataFrame(train_x)
dev_x = pd.DataFrame(dev_x)

logging.info("dev dataset has %d positive sample, %d negtive sample" % (len(dev_x[dev_x[2] == 1]), len(dev_x[dev_x[2] == 0])))

train_pos = train_x[train_x[2] == 1]
train_neg = train_x[train_x[2] == 0]

logging.info("train dataset has %d positive sample, %d negtive sample" % (len(train_pos), len(train_pos)))
train_x = train_pos.append(train_neg[:len(train_pos)])

train_x.to_csv(os.path.join(args.data_path, "train.csv"), index = False)
dev_x.to_csv(os.path.join(args.data_path,"dev.csv"), index = False)




# ------------------------  buding datasets ------------------------
logging.info("Building datasets, loding word vectors")
TEXT = SplitReversibleField(sequential=True, tokenize='spacy',lower=args.lower, include_lengths=True,use_vocab=True,batch_first=True)
LABEL = data.Field(sequential=False,use_vocab=False,is_target=True,batch_first=True)
train, dev = data.TabularDataset.splits(
        skip_header=True,
        path=args.data_path,
        train='train.csv',
        validation='dev.csv',
        format='csv',
        fields=[('qid',None),('question_text', TEXT), ('label', LABEL)])

train_iter, dev_iter = data.Iterator.splits(
        (train, dev), sort_key=lambda x: len(x.question_text),
        batch_sizes=(args.batch_size, 512),
        sort_within_batch=True,
        repeat=False,
        device=device)


# ------------------------  buding vocab ------------------------
logging.info("loading vocab")
TEXT.build_vocab(train, vectors="glove.6B.100d")

# ------------------------  buding network ------------------------
logging.info("building network")
model = Classifier(args,TEXT.vocab)
model.to(device)
criterion = nn.NLLLoss()
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
        n_correct += (torch.max(predict, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        n_total += batch.batch_size
        train_acc = 100. * n_correct/n_total
        loss = criterion(predict, batch.label)
        loss.backward(); opt.step()

        if iterations % args.dev_every == 0:
            model.eval()
            dev_iter.init_epoch()
            n_dev_correct, dev_loss = 0, 0
            with torch.no_grad():
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                     predict = model(dev_batch)
                     n_dev_correct += (torch.max(predict, 1)[1].view(dev_batch.label.size()) == dev_batch.label).sum().item()
                     dev_loss = criterion(predict, dev_batch.label)
            dev_acc = 100. * n_dev_correct / len(dev)
            print(dev_log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.item(), dev_loss.item(), train_acc, dev_acc))
        elif iterations % args.log_every == 0:
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.item(), ' '*8, n_correct/n_total*100, ' '*12))
