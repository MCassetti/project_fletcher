import itertools
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
import time


class myLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, batch_size, num_layers, vocab_dim, batch_first=True):
        super(myLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.vocab_dim = vocab_dim
        self.batch_size = batch_size
        self.mini_batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_dim, embedding_dim)
        #self.weight = Parameter(torch.Tensor(target_size, embedding_dim))
        # The linear layer that maps from hidden state space to tag space
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=batch_first)
        self.hidden2target = nn.Linear(hidden_dim, vocab_dim)
        self.hidden = self.init_hidden(batch_size)


    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))


    def forward(self, sentence, batch_size):

        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds, self.hidden)
        lstm_out = lstm_out.contiguous()
        lstm_out_flat = lstm_out.view(sentence.size(0) * sentence.size(1) , -1)
        target_space = self.hidden2target(lstm_out_flat)
        target_scores = F.log_softmax(target_space, dim=1)
        return target_scores.view(sentence.size(0), sentence.size(1), -1)


class myTrain(Dataset):
    def __init__(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
    def __len__(self):
        return len(self.X_train)
    def __getitem__(self,index):
        return self.X_train[index], self.y_train[index]

def prepare_sequence(seq, word_to_index):
    idxs = [word_to_index[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

if __name__ == "__main__":

    with open('training_v2.pkl','rb') as fp:
        training_set = pickle.load(fp)

    with open('word_dict.pkl','rb') as fp:
        word_dict = pickle.load(fp)

    index_to_word = word_dict[0]
    word_to_index = word_dict[1]
    X_train = [prepare_sequence(sent,word_to_index) for sent in training_set[0]]
    y_train = [prepare_sequence(sent,word_to_index) for sent in training_set[1]]

    dataset = myTrain(X_train,y_train)
    data_loader = DataLoader(dataset,32,shuffle=True)
    training_set = []


    hidden_dim = 256
    vocab_dim =  28669
    batch_size = 32
    num_layers = 2
    embedding_dim = 99

    model = myLSTM(embedding_dim, hidden_dim, batch_size, num_layers, vocab_dim)

    loss_function = nn.NLLLoss()
    optimizer = optim.RMSprop(model.parameters())
    #optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_over_time = []
    ts = time.time()

    print("Training LSTM at timestamp:",ts)
    for epoch in range(10):  # again, normally you would NOT do 100 epochs, it is toy data
        for index, (batch_X, batch_y) in enumerate(data_loader):
            # try:

            model.zero_grad()
            batch_size = batch_X.size(0)
            model.hidden = model.init_hidden(batch_size)

            # Step 2. Get our inputs ready for the network, that is, turn them into

            # Step 3. Run our forward pass.
    #             embedding_dim = batch_X.size(1)
            target_scores = model(batch_X, batch_size)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(target_scores.view(batch_size * batch_X.size(1), -1),
                                 batch_y.view(-1))

            loss.backward()
            optimizer.step()
            loss_over_time.append(loss.item())


            if (index % 100) == 0 and index !=0:
                print('you are at index:', index)
                filepath = os.getcwd()
                state = {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                        }
                torch.save(state, filepath + '/martha_model.pt')
                with open('losses.pkl','wb') as fp:
                    pickle.dump(loss_over_time,fp)
            # except:
            #     print('batch data inconsistent at index:', index)
            #     print('saving the current state and attempt to continue')
            #     print(batch_X.size())
            #     filepath = os.getcwd()
            #     state = {
            #             'epoch': epoch,
            #             'state_dict': model.state_dict(),
            #             'optimizer': optimizer.state_dict()
            #         }
            #     torch.save(state, filepath + '/martha_model.pt')
            #     continue

        print("The current epoch is:", epoch)
        print("Time since last epoch is",ts - time.time())


    filepath = os.getcwd()
    print('congrats, you have trained!! saving..')
    print(filepath + '/martha_model.pt')
    torch.save(model.state_dict(), filepath + '/martha_model.pt')

