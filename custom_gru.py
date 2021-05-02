import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import RNNCellBase, Parameter


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

    def foo(self, ht, a, E):
        ot = torch.rand_like(ht)
        a = torch.rand_like(a)

        return ot, a

    def attention(self, ht, a, E):
        #ref_type = nn.Softmax()(self.beta * self.S(ht))
        #print(ref_type.shape)
        # E_new = (torch.ones(a.size()[1]) - a)
        # E_new = torch.einsum('mi,mj->mij', E_new, torch.ones(E.size()[0], self.k))
        # E_new = torch.einsum('mij,mij->mij', E_new, E)
        # c_new = E_new.view(self.k, self.L) @ alpha_new
        # print(E_new.shape)
        # print(alpha_new.shape, E_new.shape)
        # print(alpha_new.shape)
        # print(c_new.shape)
        # E_used = a
        # E_used = torch.einsum('mi,mj->mij', E_used, torch.ones(self.k))
        # E_used = torch.einsum('mij,mij->mij', E_used, E)
        # print(E_used.shape)
        # alpha_used = self.P(ht)
        # alpha_used = self.gammaE_used @ alpha_used
        # alpha_used = nn.Softmax()(alpha_used)
        # print(alpha_used.shape)
        # c_used = E_used.view(self.k, self.L) @ alpha_used
        # print(c_used.shape)
        # c_gru = self.P(ht)
        # print(c_gru.shape)
        #
        # a = a + ref_type[1] * alpha_new

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
        a = torch.zeros(inp.shape[0], L)

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


m = 32
l = 10
k = 100
SEQ_LENGTH = 50
INPUT_DIM = 300

inp = torch.rand(m, SEQ_LENGTH, INPUT_DIM)
E = torch.rand(m, l, k)
g = torch.rand(m, k)


layer = CustomChecklistCell(INPUT_DIM, k)
out = layer.forward(inp, g, E)

print(out.shape)