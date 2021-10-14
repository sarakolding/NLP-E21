"""
Logistic regression implemented using the nn.module class
"""

import torch
import torch.nn as nn
from sklearn import datasets
from collections import Counter

class Model(nn.Module):
    def __init__(self, n_input_features = 10):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(n_input_features, 30)
        self.linear2 = nn.Linear(30, 30)
        self.linear3 = nn.Linear(30, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        y_pred = torch.sigmoid(x)
        return y_pred

def term_freq(tokens) -> dict: 
    """
    Takes in a spacy doc object and returns a dictionary of term frequency for each token.
    """
    # count how many times unique tokens appear in the doc
    counts = Counter([token.text for token in tokens])
    # return a dictionary with token as key and frequency as value
    return {token: count/len(tokens) for (token, count) in counts.items()}

# Create dataset
from datasets import load_dataset
dataset = load_dataset("emotion")

import spacy
nlp = spacy.load("en_core_web_sm")


train = dataset['train']
train_text = train['text']
docs = [nlp(t) for t in train_text]

freqs = [term_freq(t) for t in docs]

from sklearn.feature_extraction import DictVectorizer

v = DictVectorizer(sparse=False)
X_numpy = v.fit_transform(freqs)

y_numpy = train['label']

#X_numpy, y_numpy = datasets.make_classification(n_samples=1000, n_features=10, random_state=7)
X = torch.tensor(X_numpy, dtype=torch.float)
y = torch.tensor(y_numpy, dtype=torch.float)
y = y.view(y.shape[0], 1)
print(X)
print(y)

# initialize model
model = Model(n_input_features=10)

# define loss and optimizer
#criterion = nn.BCELoss()
#optimizer = torch.optim.Adam(model.parameters()) 

# train
#epochs = 10000
#for epoch in range(epochs):
    # forward
#    y_hat = model(X)

    # backward
#    loss = criterion(y_hat, y)
#    loss.backward()
#    optimizer.step()
#    optimizer.zero_grad()

    # some print to see that it is running
#    if (epoch+1) % 1000 == 0:
#        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
