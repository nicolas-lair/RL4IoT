import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from architecture.utils import differentiable_or


class RecurrentDifferentiableOR(nn.Module):
    def __init__(self, input_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=1, batch_first=True)

    def forward(self, proba_tensor):
        _, (h, c) = self.lstm(proba_tensor.view(proba_tensor.size(0), proba_tensor.size(2), proba_tensor.size(1)))
        return h

class DiffOR(nn.Module):
    def __init__(self, input_size=3, hidden_size=256):
        super().__init__()
        self.hidden = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, proba_tensor):
        x =self.hidden(proba_tensor)
        x = F.relu(x)
        x = self.linear(x)
        x = F.sigmoid(x)
        return x

class DumbModel(nn.Module):
    def __init__(self, in_features, or_function):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=1)
        self.or_function = or_function

    def forward(self, input):
        x = self.linear(input)
        x = nn.functional.sigmoid(x)
        x = self.or_function(x)
        return x

class DeepSetModel(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, input):
            x = self.linear1(input)
            x = F.relu(x)
            x = x.mean(dim=1)
            x = self.linear2(x)
            x = F.sigmoid(x)
            return x

def batch(batch_size, n_features, min_len, max_len, max_nb, instruction='above_mean'):
    n_obj = torch.randint(low=min_len,
                          high=max_len + 1,
                          size=(1,))

    if instruction == 'above_mean':
        x = torch.randint(high=max_nb, size=(batch_size, n_features, n_obj))
        threshold = max_nb * n_features * 0.5
        y = (x.sum(dim=1) > threshold).any(dim=1)
    elif instruction == 'simple_or':
        x = torch.rand(size=(batch_size, n_features, n_obj))
        y = (x > 0.5).any(dim=2)
    else:
        raise NotImplementedError
    return x.float(), y.float()


def train(model, optimizer, loss, task, epoch, step_per_epoch, batch_size, n_features, len_range, max_nb,
          clip=1.,
          cuda=torch.cuda.is_available()):
    """Train single epoch"""
    print('Epoch [{}] -- Train'.format(epoch))
    loss_record = []
    model.train()
    if cuda:
        model.cuda()

    for step in range(step_per_epoch):
        optimizer.zero_grad()
        x, y = batch(batch_size, n_features, min_len=len_range[0], max_len=len_range[1], max_nb=max_nb,
                     instruction=task)
        if cuda:
            x, y = x.cuda(), y.cuda()
        if task == 'above_mean':
            out = model(x.view(batch_size, -1, n_features))
        else:
            out = model(x)

        # bool_reward = or_function(obj_proba).view(-1)

        loss_tensor = loss(out, y)

        # Backward
        loss_tensor.backward()
        # TODO check if necessary
        # nn.utils.clip_grad_norm_(policy_network.parameters(), clip)
        optimizer.step()

        if (step + 1) % 100 == 0:
            print('Epoch [{}] loss: {}'.format(epoch, loss.item()))
        return np.mean(loss_record)


@torch.no_grad()
def evaluate(model, task, epoch, eval_batch_size, n_features, len_range, max_nb, n_eval, verbose=True):
    """Evaluate after a train epoch"""
    print('Epoch [{}] -- Evaluate'.format(epoch))

    cuda = next(model.parameters()).is_cuda
    if cuda:
        model.cpu()

    accuracy = []
    # min_max_accuracy = []
    for i in range(n_eval):
        x_val, y_val = batch(eval_batch_size, n_features, min_len=len_range[0], max_len=len_range[1], max_nb=max_nb,
                             instruction=task)

        if task == 'above_mean':
            out = model(x_val.view(eval_batch_size, -1, n_features))
        else:
            out = model(x_val)
        # bool_reward = or_function(obj_proba).view(-1)
        bool_reward = out >= 0.5

        accuracy.append((y_val == bool_reward).float().mean().item())
        # min_max_accuracy.append((x_val[:, y_val] == x_val[:, prediction]).float().mean().item())

    if verbose:
        print(f'idx accuracy: {round(np.mean(accuracy), 2), round(np.std(accuracy), 2)}')
        # print(f'min_max accuracy: {round(np.mean(min_max_accuracy), 2), round(np.std(min_max_accuracy), 2)}')

    if cuda:
        model.cuda()
    return accuracy


if __name__ == "__main__":
    CUDA = True

    HIDDEN_SIZE = 256
    IN_FEATURES = 10
    BATCH_SIZE = 64
    LEN_RANGE = (3, 3)
    MAX_NB = 10

    EVAL_BATCH_SIZE = 25
    N_EVAL = 25

    STEPS_PER_EPOCH = 250
    EPOCHS = 400
    loss = nn.BCELoss()

    OR_FUNC = differentiable_or
    # TASK = 'simple_or' # 'above_mean' # 'simple_or'


    ############# SIMPLE OR #######################
    TASK = 'simple_or' # 'above_mean' # 'simple_or'
    N_OBJ = 3

    net = RecurrentDifferentiableOR(1)
    # net = DiffOR(N_OBJ)
    optimizer = optim.Adam(net.parameters())
    loss = nn.BCELoss()
    for epoch in range(EPOCHS):
        import time

        t0 = time.time()
        loss_epoc = train(net,
                          optimizer=optimizer,
                          loss=loss,
                          task=TASK,
                          epoch=epoch + 1,
                          step_per_epoch=STEPS_PER_EPOCH,
                          batch_size=BATCH_SIZE,
                          n_features=1,
                          len_range=(N_OBJ, N_OBJ),
                          max_nb=MAX_NB,
                          cuda=CUDA)
        print('{} seconds'.format(time.time() - t0))
        acc = evaluate(net,
                       task=TASK,
                       epoch=epoch + 1,
                       eval_batch_size=EVAL_BATCH_SIZE,
                       n_features=1,
                       len_range=(N_OBJ, N_OBJ),
                       max_nb=MAX_NB,
                       n_eval=N_EVAL)

    #
    # network = DumbModel(in_features=IN_FEATURES, or_function=OR_FUNC)
    # network2 = DeepSetModel(in_features=IN_FEATURES, hidden_size=256)
    #
    # optimizer = optim.Adam(network.parameters())
    # idx_acc = []
    # loss_record = []
    # for epoch in range(EPOCHS):
    #     import time
    #
    #     t0 = time.time()
    #     loss_epoc = train(network,
    #                  optimizer=optimizer,
    #                  loss=loss,
    #                  task=TASK,
    #                  epoch=epoch + 1,
    #                  step_per_epoch=STEPS_PER_EPOCH,
    #                  batch_size=BATCH_SIZE,
    #                  n_features=IN_FEATURES,
    #                  len_range=LEN_RANGE,
    #                  max_nb=MAX_NB,
    #                  cuda=CUDA)
    #     print('{} seconds'.format(time.time() - t0))
    #     acc = evaluate(network,
    #                    task=TASK,
    #                    epoch=epoch + 1,
    #                    eval_batch_size=EVAL_BATCH_SIZE,
    #                    n_features=IN_FEATURES,
    #                    len_range=LEN_RANGE,
    #                    max_nb=MAX_NB,
    #                    n_eval=N_EVAL)
    #     #     min_max_acc.append(min_max)
    #
    #     loss_record.append(loss_epoc)
    #     idx_acc.append(acc)
