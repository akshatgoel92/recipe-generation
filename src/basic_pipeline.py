# +
import numpy as np
import pandas as pd
import torch
import re
import fasttext
from tensorflow.keras.preprocessing.sequence import pad_sequences


import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import RNNCellBase, Parameter


# +
def process_str_lst(x):
    lst = eval(x)
    return ' . '.join(lst)


def process_str_lst2(x):
    lst = eval(x)
    return lst


def tokenize_sentence(sentence, word2idx, sent_type='goal'):
    unk_index = word2idx['UNK']
    sos_index = word2idx['SOS']
    eos_index = word2idx['EOS']
    
    tokenized_sentences = []
    
    list_of_tokens = []
    if sent_type == 'recipe':
        list_of_tokens.append(sos_index)
        
    sentence = sentence.split(' ') if type(sentence) is type('x') else sentence
    
    for word in sentence:
        if word in word2idx:
            list_of_tokens.append(word2idx[word])
        else:
            list_of_tokens.append(unk_index)
    
    if sent_type == 'recipe':
        list_of_tokens.append(eos_index)
    
    return list_of_tokens


def train_test_split(df, train_percent=0.8):
    '''
    --------------------------
    Split data into train and test sets
    Input: Dataframe containing assignment data
    Output: Training dataframe and testing dataframe
    split according to specified percentage
    -------------------------
    '''
    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Calculate no. of training examples based on train_percent
    # Here we use 2/3, 1/3 by default as required by the assignment
    n_train = round(train_percent*len(df))
    
    # Filter the dataframe to get training and testing rows
    df_train = df[:n_train]
    df_test = df[n_train:]
    
    df_val = df_test[:len(df_test)//2].reset_index(drop=True)
    df_test = df_test[len(df_test)//2:].reset_index(drop=True)
    
    return df_train, df_val, df_test


def create_input(tokenized_sent):
    
    inp = tokenized_sent[:-1]
    
    return inp


def create_output(tokenized_sent):
    out = tokenized_sent[1:]
    
    return out    


def sent_length(sent):
    return len(sent.split())


def preprocess(sent):

    if type(sent) == type([]):
        sent = [re.sub(r'[\d]+', '', word) for word in sent]
    else:
        sent = re.sub(r'[\d]+', '', sent)
    
    return sent


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


def clear_gpu(model):
    # Removes model from gpu and clears the memory
    model = model.to('cpu')
    del model
    torch.cuda.empty_cache()


# -

ft = fasttext.load_model('cc.en.300.bin')

df = pd.read_csv('../dataset/RAW_recipes.csv')
df = df.drop(['id', 'contributor_id', 'minutes', 'description', 'submitted', 'tags', 'nutrition', 'n_steps', 'n_ingredients'], axis=1)
df = df.dropna()

df['instructions'] = df['steps'].apply(process_str_lst)
#df['instructions'] = df['instructions'].apply(preprocess)
df['name'] = df['name'].apply(preprocess)
df['ingredients'] = df['ingredients'].apply(preprocess)
df['recipe_length'] = df['instructions'].apply(sent_length)

# +
#### TEST CASE FOR TOKENIZER

# x = 'tomato pasta'
# ingr = ['olive oil', 'tomato']

# for i in tokenize_sentence(x):
#     print(idx2word[i])

# print('\n')
# for i in tokenize_sentence(ingr):
#     print(idx2word[i])
# -

# # Trim Recipes

# +
# 95% of the recipes are under 250 words (TG!)

import seaborn as sns

sns.set_style('darkgrid')
sns.displot(df['recipe_length'].values)
sum(df['recipe_length'] <= 250)/len(df)
# -

df = df[df['recipe_length'] <= 250].reset_index(drop=True)

# +
# df['tokenized_instructions'] = df['instructions'].apply(tokenize_sentence)
# df['inp'] = df['tokenized_instructions'].apply(create_input)
# df['out'] = df['tokenized_instructions'].apply(create_output)
# -

# # Vocab Creation

# +
from collections import Counter

c = 0
ingredients = list()

for lst in df['ingredients'].values:
    for ingredient in eval(lst):
        ingredients.append(ingredient)

counts = Counter(ingredients)
ingredient_dict = {k: v for k, v in counts.items() if v > 10}

len(ingredient_dict)

# +
words = []

for string in df['name'].values:
    if (string is not None) and (string is not np.nan):
        for word in string.split():
            words.append(word)

for string in df['instructions'].values:
    if (string is not None) and (string is not np.nan):
        for word in string.split():
            words.append(word)
            
counts = Counter(words)
word_dict = {k: v for k, v in counts.items() if v > 50}

print(len(word_dict))

# +
# Combine Ingredient Vocab and Word Vocab

lst = list(word_dict.keys()) + list(ingredient_dict.keys())
vocab = set(lst)

print(len(vocab))


# -

# # Emb Matrix Creation

def get_fasttext_embedding_matrix(token_list):
    word_to_number = {}
    number_to_word = {}
    emb_list = []
    c = 1
    emb_list.append(np.zeros(ft.get_dimension()))
    token_list = token_list + ['UNK','EOS','SOS']
    for token in token_list :
        emb = ft.get_word_vector(token)
        emb_list.append(emb)
        word_to_number[token] = c 
        number_to_word[c] = token
        c = c+1
    emb_mat = np.vstack(emb_list)
    return emb_mat,word_to_number,number_to_word


emb_mat, word2idx, idx2word =  get_fasttext_embedding_matrix(list(vocab))
df['tokenized_instructions'] = df['instructions'].apply(lambda x: tokenize_sentence(x, word2idx))
df['tokenized_goal'] = df['name'].apply(lambda x: tokenize_sentence(x, word2idx, sent_type='recipe'))
df['ingredients'] = df['ingredients'].apply(process_str_lst2)
df['tokenized_ingredients'] = df['ingredients'].apply(lambda x: tokenize_sentence(x, word2idx))

# # DataLoader

df_train, df_val, df_test = train_test_split(df)

goal_train = pad_sequences(df_train['tokenized_goal'])
recipe_train = pad_sequences(df_train['tokenized_instructions'])
ingr_train = pad_sequences(df_train['tokenized_ingredients'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_EPOCH = 3
# -

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


