
import numpy as np
import pandas as pd
import json

import torch
import torch.nn.functional as F
from torch.nn import RNNCellBase, Parameter


def clear_gpu(model):
    # Removes model from gpu and clears the memory
    model = model.to('cpu')
    del model
    torch.cuda.empty_cache()


class Dataset(torch.utils.data.Dataset):
    # Basic dataset class to work with torch data loader

    def __init__(self, goal, recipe, ingredients):
        self.X = recipe[:, :-1]
        self.y = recipe[:, 1:]
        self.goal = goal
        self.ingredients = ingredients

        assert len(self.X) == len(self.y), print("Number of examples don't match up")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.goal[index], self.ingredients[index]


class BasicModel(torch.nn.Module):
    def __init__(self, wv_matrix):
        super(BasicModel, self).__init__()
        vocab_size, embedding_size = wv_matrix.shape

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data.copy_(torch.from_numpy(wv_matrix))
        self.embedding.weight.requires_grad = False

        self.lstm = torch.nn.LSTM(
            input_size=embedding_size,
            hidden_size=64,
            dropout=0.2,
        )

        self.fc = torch.nn.Linear(64, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        output, state = self.lstm(embed)
        logits = self.fc(output)
        return logits


class Model(torch.nn.Module):
    def __init__(self, wv_matrix):
        super(Model, self).__init__()
        vocab_size, embedding_size = wv_matrix.shape
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data.copy_(torch.from_numpy(wv_matrix))
        self.embedding.weight.requires_grad = False
        hidden_size = embedding_size
        self.cgru = CustomChecklistCell(embedding_size, hidden_size)
        self.fc = torch.nn.Linear(embedding_size, vocab_size)

    def forward(self, recipe, g, ingr):
        recipe_embed = self.embedding(recipe)
        goal_embed = self.embedding(g)
        ingr_embed = self.embedding(ingr)
        goal_embed = goal_embed.sum(axis=1)
        output = self.cgru(recipe_embed, goal_embed, ingr_embed)
        logits = self.fc(output)
        return logits


class CustomChecklistCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, batch_first=True, beta=5, gamma=2):
        super(CustomChecklistCell, self).__init__(input_size, hidden_size, bias, num_chunks=5)

        self.batch_first = batch_first

        self.Z = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.Y = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_g = Parameter(torch.Tensor(hidden_size, hidden_size))

        self.z_bias = Parameter(torch.ones(hidden_size))
        self.y_bias = Parameter(torch.ones(hidden_size))

        self.beta = beta
        self.gamma = gamma

        self.S = Parameter(torch.ones(3, hidden_size))
        self.P = Parameter(torch.ones(hidden_size, hidden_size))

    def init_hidden(self, g):
        return F.linear(g, self.U_g)

    def cell(self, inp, hidden, g, E_t_new, activation=F.tanh):

        w_ih = self.weight_ih
        w_hh = self.weight_hh
        b_ih = self.bias_ih
        b_hh = self.bias_hh
        Y = self.Y
        Z = self.Z
        y_bias = self.y_bias
        z_bias = self.z_bias

        # Calculate all W @ inp. w_ih is actually 5 different weights
        gi = F.linear(inp, w_ih, b_ih)

        # Calcualtes all W @ hidden, w_hh is actually 5 different weights
        gh = F.linear(hidden, w_hh, b_hh)

        # Split them into seperate terms
        i_reset, i_update, i_new, i_goal, i_item = gi.chunk(5, 1)
        h_reset, h_update, h_new, h_goal, h_item = gh.chunk(5, 1)

        # Update Gate
        z_t = F.sigmoid(i_update + h_update)

        # Reset Gate
        r_t = F.sigmoid(i_reset + h_reset)

        # Goal Select Gate
        s_t = F.sigmoid(i_goal + h_goal)

        # Item Select Gate
        q_t = F.sigmoid(i_item + h_item)

        # That term
        tmp = torch.einsum('mlk -> mk', E_t_new)

        # New Gate
        h_tilde_t = activation(i_new
                               + r_t * h_new
                               + s_t * F.linear(g, Y, y_bias)
                               + q_t * F.linear(tmp, Z, z_bias))

        # tp1: t plus 1
        h_tp1 = h_tilde_t + z_t * (hidden - h_tilde_t)

        return h_tp1

    def attention(self, ht, a, E):
        ref_type = torch.nn.Softmax()(F.linear(self.beta * ht, self.S))
        h_proj = F.linear(ht, self.P)

        E_new = torch.einsum('ml, mlk -> mlk', 1 - a, E)
        alpha_new = self.gamma * torch.einsum('mk,mlk->ml', h_proj, E_new)
        alpha_new = torch.nn.Softmax()(alpha_new)
        c_new = torch.einsum('mlk, ml -> mk', E, alpha_new)

        E_used = torch.einsum('ml, mlk -> mlk', a, E)
        alpha_used = self.gamma * torch.einsum('mk,mlk->ml', h_proj, E_used)
        alpha_used = torch.nn.Softmax()(alpha_used)
        c_used = torch.einsum('mlk, ml -> mk', E, alpha_used)

        c_gru = h_proj

        out = ref_type[:, 0].reshape(-1, 1)*c_gru + ref_type[:, 1].reshape(-1, 1)*c_new + ref_type[:, 2].reshape(-1, 1)*c_used
        a = a + ref_type[:, 1].reshape(-1, 1) * alpha_new

        return out, a, E_new

    def forward(self, inp, g, E):

        # E will need padding as well so that it can be minibatched
        '''
        inp: (examples, seq_length, inp_dim) assuming batch_first
        g: (examples, hidden_dim)
        E: (examples, agenda length, hidden_dim)
        '''
        L = E.shape[1]

        ht = self.init_hidden(g)
        a = torch.zeros(inp.shape[0], L).to(device)

        if len(inp.shape) == 3:
            if self.batch_first:
                inp = inp.transpose(0, 1)

        E_t_new = torch.einsum('ml, mlk -> mlk', 1-a, E)

        lst_o = list()
        for t in np.arange(inp.shape[0]):

            ht = self.cell(
                inp[t], ht, g, E_t_new
            )

            ot, a, E_t_new = self.attention(ht, a, E)
            lst_o.append(ot)

        output = torch.stack(lst_o)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('../dataset/RAW_recipes.csv')
goal_train = np.load('goal.npy')
recipe_train = np.load('recipe.npy')
ingr_train = np.load('ingr.npy')
emb_mat = np.load('emb_mat.npy')

a_file = open("word2idx.json", "r")
word2idx = json.load(a_file)
a_file.close()

a_file = open("idx2word.json", "r")
idx2word = json.load(a_file)
a_file.close()

MAX_EPOCH = 3

dataloader_params = {'batch_size': 64, 'shuffle': True, 'num_workers': 6}
train_data = Dataset(goal_train, recipe_train, ingr_train)
train_generator = torch.utils.data.DataLoader(train_data, **dataloader_params)

model = Model(emb_mat).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 1
clip = 10
for epoch in range(epochs):
    # Iterates through minibatches and does updates to weights
    running_loss = 0
    for data in train_generator:
        recipe, label, goal, ingr = data
        label = label.reshape(-1)
        recipe, label = recipe.type(torch.LongTensor).to(device), label.type(torch.LongTensor).to(device)
        goal, ingr = goal.type(torch.LongTensor).to(device), ingr.type(torch.LongTensor).to(device)

        optimizer.zero_grad()
        outputs = model.forward(recipe, goal, ingr)

        i, j, k = outputs.shape
        outputs = outputs.reshape(i*j, k)
        loss = loss_fn(outputs, label)
        loss.backward()

        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        running_loss += loss.item()

    running_loss /= len(train_generator)
    print("epoch: ",epoch, "train_loss: ",running_loss)


