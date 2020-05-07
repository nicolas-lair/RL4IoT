"""
Module implementing the pointer network proposed at: https://arxiv.org/abs/1506.03134
The implementation try to follows the variables naming conventions
ei: Encoder hidden state
di: Decoder hidden state
di_prime: Attention aware decoder state
W1: Learnable matrix (Attention layer)
W2: Learnable matrix (Attention layer)
V: Learnable parameter (Attention layer)
uj: Energy vector (Attention Layer)
aj: Attention mask over the input
"""

import time

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""
Generate random data for pointer network
"""
import torch
from torch.utils.data import Dataset


def batch(batch_size, min_len, max_len, max_nb, instruction='min_max'):
    array_len = torch.randint(low=min_len,
                              high=max_len + 1,
                              size=(1,))

    x = torch.randint(high=max_nb, size=(batch_size, array_len))
    if instruction == 'sort':
        y = x.argsort(dim=1)
        idx = None
    elif instruction == 'min_max':
        idx = torch.randint(low=0, high=2, size=(batch_size,)).bool()
        y = torch.zeros(batch_size, dtype=torch.long)
        y[~idx] = x.argmin(dim=1)[~idx]
        y[idx] = x.argmax(dim=1)[idx]
        idx = idx.long()
    elif 'many' in instruction:
        n_instr = int(instruction.split('_')[1])
        idx = torch.randint(low=0, high=n_instr, size=(batch_size,), dtype=torch.long)
        y = torch.zeros(batch_size, dtype=torch.long)
        t = x.argsort(dim=1)

        for i in range(n_instr):
            mask = idx == i
            func = (i % 2 == 0) * (i // 2) - (i % 2 == 1) * (i // 2 + 1)
            y[mask] = t[mask, func]
    else:
        raise NotImplementedError
    return x, y, idx


class Encoder(nn.Module):
    def __init__(self, hidden_size, bidirectional=False, n_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True, bidirectional=bidirectional, num_layers=n_layers)
        self.bidirectional = bidirectional

    def forward(self, x):
        # x: (BATCH, ARRAY_LEN, 1)
        return self.lstm(x.view(*x.size(), 1))


class Attention(nn.Module):
    def __init__(self, hidden_size, units):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_size, units, bias=False)
        self.W2 = nn.Linear(hidden_size, units, bias=False)
        self.V = nn.Linear(units, 1, bias=False)

    def forward(self,
                encoder_out,
                instruction_embedding):
        # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
        # decoder_hidden: (BATCH, HIDDEN_SIZE)

        # Add time axis to decoder hidden state
        # in order to make operations compatible with encoder_out
        # instruction_embedding: (BATCH, 1, HIDDEN_SIZE)
        instruction_embedding = instruction_embedding.unsqueeze(1)

        # uj: (BATCH, ARRAY_LEN, ATTENTION_UNITS)
        # Note: we can add the both linear outputs thanks to broadcasting
        uj = self.W1(encoder_out) + self.W2(instruction_embedding)
        uj = torch.tanh(uj)

        # uj: (BATCH, ARRAY_LEN, 1)
        uj = self.V(uj)

        # Attention mask over inputs
        # aj: (BATCH, ARRAY_LEN, 1)
        # aj = F.softmax(uj, dim=1)

        # di_prime: (BATCH, HIDDEN_SIZE)
        # di_prime = aj * encoder_out
        # di_prime = di_prime.sum(1)

        # return di_prime, uj.squeeze(-1)

        return uj.squeeze(-1)


class InstructedPointerNetwork(nn.Module):
    def __init__(self, hidden_size, units, n_instruction=2, bidirectional_encoder=False, n_encoder_layers=1):
        super().__init__()
        self.encoder = Encoder(hidden_size=hidden_size, bidirectional=bidirectional_encoder, n_layers=n_encoder_layers)
        self.attention = Attention(hidden_size=(1 + int(bidirectional_encoder)) * hidden_size, units=units)
        self.instruction_embedding = nn.Embedding(num_embeddings=n_instruction,
                                                  embedding_dim=(1 + int(bidirectional_encoder)) * hidden_size)

    def forward(self, sequence, instruction):
        embedded_instruction = self.instruction_embedding(instruction)
        output, _ = self.encoder(sequence.float())
        # if self.encoder.bidirectional:
        #     output = output.view(output.size(0), output.size(1), 2, output.size(2) // 2)
        #     h0 = output[:, 0, 1, :]
        #     hn = output[:, -1, 0, :]
        #     output = torch.cat([h0, hn], dim=-1)
        attn_vector = self.attention(output, embedded_instruction)
        seq_proba = attn_vector.squeeze(-1)
        prediction = seq_proba.softmax(1).argmax(1)
        return seq_proba, prediction


def train(model, optimizer, task, epoch, step_per_epoch, batch_size, len_range, max_nb, clip=1.,
          cuda=torch.cuda.is_available()):
    """Train single epoch"""
    print('Epoch [{}] -- Train'.format(epoch))
    loss_record = []
    model.train()
    if cuda:
        model.cuda()

    for step in range(step_per_epoch):
        optimizer.zero_grad()
        x, y, instruction = batch(batch_size, min_len=len_range[0], max_len=len_range[1], max_nb=max_nb,
                                  instruction=task)
        if cuda:
            x, y, instruction = x.cuda(), y.cuda(), instruction.cuda()
        seq_proba, _ = model(sequence=x, instruction=instruction)

        # Forward
        # x, y = batch(BATCH_SIZE)
        # seq_proba, prediction = model(x, y)
        loss = F.cross_entropy(seq_proba, y)

        # loss_record.append(loss.mean().item())

        # Backward
        loss.backward()
        # TODO check if necessary
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if (step + 1) % 100 == 0:
            print('Epoch [{}] loss: {}'.format(epoch, loss.item()))
        # return np.mean(loss_record)


@torch.no_grad()
def evaluate(model, task, epoch, eval_batch_size, len_range, max_nb, n_eval, verbose=True):
    import numpy as np
    """Evaluate after a train epoch"""
    print('Epoch [{}] -- Evaluate'.format(epoch))

    cuda = next(model.parameters()).is_cuda
    if cuda:
        model.cpu()

    idx_accuracy = []
    # min_max_accuracy = []
    for i in range(n_eval):
        x_val, y_val, instruction = batch(eval_batch_size, min_len=len_range[0], max_len=len_range[1], max_nb=max_nb,
                                          instruction=task)

        _, prediction = model(sequence=x_val, instruction=instruction)
        idx_accuracy.append((y_val == prediction).float().mean().item())
        # min_max_accuracy.append((x_val[:, y_val] == x_val[:, prediction]).float().mean().item())

    if verbose:
        print(f'idx accuracy: {round(np.mean(idx_accuracy), 2), round(np.std(idx_accuracy), 2)}')
        # print(f'min_max accuracy: {round(np.mean(min_max_accuracy), 2), round(np.std(min_max_accuracy), 2)}')

    if cuda:
        model.cuda()
    return idx_accuracy

    # for i in range(out.size(0)):
    #     print('{} --> {} --> {}'.format(
    #         x_val[i],
    #         x_val[i].gather(0, out[i]),
    #         x_val[i].gather(0, y_val[i])
    #     ))


if __name__ == "__main__":
    CUDA = True
    HIDDEN_SIZE = 256
    ATTENTION_UNITS = 128

    BATCH_SIZE = 64
    LEN_RANGE = (5, 15)
    MAX_NB = 100

    STEPS_PER_EPOCH = 250
    EPOCHS = 100

    N_INSTR = 5
    TASK = 'many'
    if 'many' in TASK:
        TASK += f'_{N_INSTR}'

    ptr_net = InstructedPointerNetwork(hidden_size=HIDDEN_SIZE, units=ATTENTION_UNITS, n_instruction=N_INSTR,
                                       bidirectional_encoder=True)

    optimizer = optim.Adam(ptr_net.parameters())
    # min_max_acc = []
    idx_acc = []
    loss_record = []
    for epoch in range(EPOCHS):
        t0 = time.time()
        loss = train(ptr_net, optimizer,
              task=TASK,
              epoch=epoch + 1,
              step_per_epoch=STEPS_PER_EPOCH,
              batch_size=BATCH_SIZE,
              len_range=LEN_RANGE,
              max_nb=MAX_NB,
              cuda=CUDA)
        print('{} seconds'.format(time.time() - t0))
        acc = evaluate(ptr_net,
                       task=TASK,
                       epoch=epoch + 1,
                       step_per_epoch=STEPS_PER_EPOCH,
                       batch_size=BATCH_SIZE,
                       len_range=LEN_RANGE,
                       max_nb=MAX_NB)
        #     min_max_acc.append(min_max)

        loss_record.append((loss))
        idx_acc.append(acc)
