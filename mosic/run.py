import torch, os
import copy, traceback
import pickle
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from data import Provider
from model import Alex, Inception, Directional
from tester import RMSE, MeanCorr
from collections import defaultdict
from datetime import datetime
from functools import partial
import numpy as np

'''
This file provides functions to execute experiments
run:  execute one experiment with given args, 
      then load states with best validation error for testing.
      It will write all the experiment data into data/records.db.
      If state is given, it will not train the model, 
      but use the states directly for testing.
      If index is given, the testing results will overwrite
      record in given index.
      It returns states in all epochs
runs: This function provides safety guard to run.
      It can be helpful to run many experiments without being
      interrupted by exceptions.
      Also, it stores states into data/states.db
'''

def run(model, continuous, batchsize, resolution, epochs, spec='cqt', phase=False, size=512, lr=1e-5, weight_decay=.0, state=None, index=None):
    train = Provider('train', resolution, continuous, batchsize, spec, phase)
    valid = Provider('valid', resolution, continuous, batchsize, spec, phase)
    test = Provider('test', resolution, continuous, batchsize, spec, phase)
    device = torch.device('cuda:0')
    model = model(train.shape(), 120 if continuous else 2, size)
    net = model.cuda()
    funcs = {'rmse': RMSE, "mcorr": MeanCorr} if continuous else {'rmse': RMSE}
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    testers = {i: {k: v() for k, v in funcs.items()} for i in ['aro', 'val']}

    params = 0
    for i in model.parameters():
        params += np.asarray(i.shape).prod()

    records = {
        'args': {'model': model.identifier(), 'time': datetime.now(), 'params': params},
        'time': [],
        'train': defaultdict(partial(defaultdict, list)),
        'valid': defaultdict(partial(defaultdict, list)),
        'test': defaultdict(partial(defaultdict, list))
    }
    for i in ['continuous', 'batchsize', 'resolution', 'epochs',
              'spec', 'phase', 'size', 'lr', 'weight_decay', 'params']:
        records['args'][i] = locals()[i]

    def forward(update, data, dataset):
        loss = 0
        net.train(update)
        for x, y in tqdm(data, ncols=55, desc=data.group):
            x = torch.Tensor(x).float().to(device=device)
            y = torch.Tensor(y).float().to(device=device)
            p = net.forward(x)
            l = criterion(p, y)
            if update:
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            p = p.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            loss += l.item() * y.shape[0]
            size = int(y.shape[1] / 2)
            val_p = p[:, :size]
            val_y = y[:, :size]
            aro_p = p[:, size:]
            aro_y = y[:, size:]
            for _, i in testers['aro'].items():
                i.accept(aro_p, aro_y)
            for _, i in testers['val'].items():
                i.accept(val_p, val_y)

            del x, y, p, l

        output = dataset + ': '
        for g, i in testers.items():
            for t, j in i.items():
                value = j.finish()
                output += '{}-{}: {:.3f}, '.format(g, t, value)
                records[dataset][g][t].append(value)
        loss /= data.total()
        records[dataset]['gnr']['loss'].append(loss)
        print(output + "loss: {:.4f}".format(loss))
        return loss

    train_loss = []
    valid_loss = []
    best_loss = None
    best_model = None
    states = []
    start = datetime.now()
    if state is None:
        for i in range(epochs):
            print('Epoch {}'.format(i + 1))
            train_loss.append(forward(True, train, 'train'))
            valid_loss.append(forward(False, valid, 'valid'))
            states.append(copy.deepcopy(net.state_dict()))

            if best_loss is None or valid_loss[-1] < best_loss:
                best_loss = valid_loss[-1]
                best_model = i
            print()
            records['time'].append(datetime.now() - start)
        net.load_state_dict(states[best_model])
    else:
        net.load_state_dict(state)
        states = [state]

    forward(False, test, 'test')
    if os.path.exists('data/records.db'):
        with open('data/records.db', 'rb') as f:
            data = pickle.load(f)
    else:
        data = []

    if state is None:
        data.append(records)
    elif index is not None:
        data[index]['test'] = records['test']

    with open('data/records.db', 'wb') as f:
        pickle.dump(data, f)

    if state is None:
        plt.plot(train_loss, label='train')
        plt.plot(valid_loss, label='valid')
        plt.ylim(0, 0.05)
        plt.legend()
        plt.show()

    return states


def runs(model, continuous, batchsize, resolution, epochs, spec='cqt', phase=False, size=512, lr=1e-5, weight_decay=.0, state=None, index=None):
    if state is None:
        print(locals())
    try:
        states = run(model, continuous, batchsize, resolution, epochs, spec, phase, size, lr, weight_decay, state, index)
        with open('data/states.db', 'wb') as f:
            pickle.dump(states, f)
    except Exception as e:
        print('Error, abort')
        traceback.print_exc()
