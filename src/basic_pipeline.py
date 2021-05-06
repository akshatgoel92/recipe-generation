# +
import json

import pandas as pd
import seaborn as sns
from collections import Counter
import re
import fasttext
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np



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






# -

ft = fasttext.load_model('cc.en.300.bin')

df = pd.read_csv('../dataset/RAW_recipes.csv')
df = df.drop(['id', 'contributor_id', 'minutes', 'description', 'submitted', 'tags', 'nutrition', 'n_steps', 'n_ingredients'], axis=1)
df = df.dropna()

df['instructions'] = df['steps'].apply(process_str_lst)
df['name'] = df['name'].apply(preprocess)
df['ingredients'] = df['ingredients'].apply(preprocess)
df['recipe_length'] = df['instructions'].apply(sent_length)

sns.set_style('darkgrid')
sns.displot(df['recipe_length'].values)
sum(df['recipe_length'] <= 250)/len(df)
# -

df = df[df['recipe_length'] <= 250].reset_index(drop=True)

# # Vocab Creation

# +

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


# # Saving Outputs

a_file = open("word2idx.json", "w")
json.dump(word2idx, a_file)
a_file.close()

a_file = open("idx2word.json", "w")
json.dump(idx2word, a_file)
a_file.close()

np.save('goal', goal_train)
np.save('recipe', recipe_train)
np.save('ingr', ingr_train)
np.save('emb_mat', emb_mat)
3