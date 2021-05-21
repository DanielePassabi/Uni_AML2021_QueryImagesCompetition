import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ImageClassifier():
    def __init__(self, batch_size, epochs, ilr=0.0001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.initial_learning_rate = ilr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, training_data, training_labels, test_data, test_labels, model):
        supervised_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.initial_learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        print(f'Using device: {self.device}')
        model.to(self.device)

        def train_step(data, labels):
            optimizer.zero_grad()
            data = torch.tensor(data, dtype=torch.float32).to(self.device)
            data = data.permute(0, 3, 1, 2)
            labels = torch.tensor(labels) .to(self.device)

            preds = model(data)
            loss = supervised_loss(preds, labels)

            loss.backward()
            optimizer.step()

            eq = torch.eq(labels, torch.max(preds, -1).indices)
            accuracy = torch.sum(eq).float()/labels.shape[0] * 100
            return loss.detach().cpu(), accuracy.cpu()


        def test_step(data, labels):
            data = torch.tensor(data, dtype=torch.float32).to(self.device)
            data = data.permute(0, 3, 1, 2)
            labels = torch.tensor(labels).to(self.device)
            with torch.no_grad():
                preds = model(data)
            loss = supervised_loss(preds, labels)
            return loss.cpu(), preds.cpu()

        global_step = 0
        best_accuracy = 0.0
        for e in range(self.epochs):
            model.train()
            ## Shuffling training set
            perm = np.arange(len(training_labels))
            random.shuffle(perm)
            training_data = training_data[perm]
            training_labels = training_labels[perm]

            ## Iteration
            for i in range(0, len(training_labels), self.batch_size):
                data = training_data[i:i+self.batch_size, :]
                labels = training_labels[i:i+self.batch_size, ].astype('int64')
                global_step += 1  # len(labels)
                batch_loss, batch_accuracy = train_step(data, labels)
                if global_step % 1 == 0:
                    print('[{0}-{1:03}] loss: {2:0.05}, batch_accuracy: {3:0.03}'.format(
                        e + 1, global_step,
                        batch_loss.numpy(),
                        batch_accuracy.numpy()))
                if global_step == 1:
                    print('number of model parameters {}'.format(count_parameters(model)))

            scheduler.step()
            # Test the whole test dataset
            model.eval()
            test_preds = []
            total_loss = list()
            for i in range(0, len(test_labels), self.batch_size):
                data = test_data[i:i+self.batch_size, :]
                labels = test_labels[i:i+self.batch_size, ].astype('int64')
                batch_loss, preds = test_step(data, labels)
                batch_preds = torch.max(preds, -1).indices
                test_preds.append(batch_preds)
                total_loss.append(batch_loss)

            test_preds = torch.cat(test_preds, dim=0)
            loss = sum(total_loss)/len(total_loss)
            eq = torch.eq(torch.tensor(test_labels), test_preds).sum()
            test_accuracy = torch.true_divide(eq, torch.tensor(test_labels).shape[0]) * 100
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
            print('End of Epoch {0}/{1:03} -> loss: {2:0.05}, test accuracy: {3:0.03} - best accuracy: {4:0.03}'.format(
                        e + 1, self.epochs,
                        loss.numpy(),
                        test_accuracy.numpy(),
                        best_accuracy))

    def train_pytorch(self, training_loader, test_loader, model):
        supervised_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.initial_learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

        model.to(self.device)

        def train_step(data, labels):
            optimizer.zero_grad()
            data = data.to(self.device)
            labels = labels.to(self.device)

            preds = model(data)
            loss = supervised_loss(preds, labels)

            loss.backward()
            optimizer.step()

            eq = torch.eq(labels, torch.max(preds, -1).indices)
            accuracy = torch.sum(eq).float()/labels.shape[0] * 100
            return loss.cpu().detach(), accuracy.cpu()


        def test_step(data, labels):
            data = data.to(self.device)
            labels = labels.to(self.device)
            # we deactivate torch autograd
            with torch.no_grad():
                preds = model(data)
            loss = supervised_loss(preds, labels)
            return loss.cpu(), preds.cpu()

        best_accuracy = 0.0
        for e in range(self.epochs):
            # we activate dropout, BN params
            model.train()
            ## Iteration
            for i, batch in enumerate(training_loader):
                data, labels = batch
                batch_loss, batch_accuracy = train_step(data, labels)

                if i % 1 == 0:
                    print('[{0}-{1:03}] loss: {2:0.05}, batch_accuracy: {3:0.03}'.format(
                        e + 1, i,
                        batch_loss.numpy(),
                        batch_accuracy.numpy()))
                if i == 0:
                    print('number of model parameters {}'.format(count_parameters(model)))

            # we call scheduler to decrease LR
            scheduler.step()

            # freeze BN params, deactivate dropout
            model.eval()
            # Test the whole test dataset
            test_preds = []
            test_labels = []
            total_loss = list()
            for i, batch in enumerate(test_loader):
                data, labels = batch
                batch_loss, preds = test_step(data, labels)

                batch_preds = torch.max(preds, -1).indices
                test_preds.append(batch_preds)
                test_labels.append(labels)
                total_loss.append(batch_loss)

            test_preds = torch.cat(test_preds, dim=0).view(-1)
            test_labels = torch.cat(test_labels, dim=0).view(-1)

            assert test_preds.shape[0] == test_labels.shape[0]

            loss = sum(total_loss)/len(total_loss)
            eq = torch.eq(test_labels, test_preds).sum()
            test_accuracy = (eq/test_labels.shape[0]) * 100
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
            print('End of Epoch {0}/{1:03} -> loss: {2:0.05}, test accuracy: {3:0.03} - best accuracy: {4:0.03}'.format(
                        e + 1, self.epochs,
                        loss.numpy(),
                        test_accuracy.numpy(),
                        best_accuracy))

