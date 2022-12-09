import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import pickle


def softmax(x):
    #Softmax of vector x
    exps = np.exp(x)
    expSum = exps.sum(axis=1)
    return torch.stack([exps[i] / expSum[i] for i in range(expSum.shape[0])])


def sigmoid(s):
    return 1 / (1 + torch.exp(-s))


def sigmoidDerivative(x):
    return x * (1 - x)

def calc_accuracy(labels, predictions):
    count = len(predictions)
    correct = 0
    for label, pred in zip(labels, predictions):
        if pred == label:
            correct += 1
    return (correct / count)

class Neural_Network:
    def __init__(self, input_size, output_size, hidden_size):
        # parameters
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)
        self.b1 = torch.zeros(self.hiddenSize)

        self.W2 = torch.randn(self.hiddenSize, self.outputSize)
        self.b2 = torch.zeros(self.outputSize)

    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.h = sigmoid(self.z1)
        self.z2 = torch.matmul(self.h, self.W2) + self.b2
        return softmax(self.z2)

    def backward(self, X, y, y_hat, lr=.1):
        batch_size = y.size(0)
        # Converting vector of true labels y to a one hot encoding matrix
        one_hot = [[1 if y[j] == k else 0 for k in range(y_hat.shape[1])] for j in range(y_hat.shape[0])]
        y = torch.Tensor(one_hot)
        # After some mathematical actions we get that the derivative of the loss function w.r.t softmax(z2)
        # multiplied by the derivative of softmax(z2) w.r.t z2 equals:
        dl_dz2 = y_hat - y

        dl_dh = torch.matmul(dl_dz2, torch.t(self.W2))
        dl_dz1 = dl_dh * sigmoidDerivative(self.h)

        self.W1 -= lr * torch.matmul(torch.t(X), dl_dz1)
        self.b1 -= lr * torch.matmul(torch.t(dl_dz1), torch.ones(batch_size))
        self.W2 -= lr * torch.matmul(torch.t(self.h), dl_dz2)
        self.b2 -= lr * torch.matmul(torch.t(dl_dz2), torch.ones(batch_size))

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)


if __name__ == '__main__':
    
    # Hyper Parameters
    input_size = 784
    hidden_size = 64
    output_size = 10
    num_classes = 10
    num_epochs = 10
    batch_size = 100

    # Normalize input
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3015,))])

    # MNIST Dataset (Images and Labels)
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transform,
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transform)

    # Dataset Loader (Input Pipline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    NN = Neural_Network(input_size, output_size, hidden_size)

    train_acc_list = []
    train_mean_acc_list = []
    test_acc_list = []
    test_mean_acc_list = []
    
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.view(100, 28 * 28)
            NN.train(images, labels)
            out = NN.forward(images)
            predictions = torch.argmax(out, 1)
            train_acc_list.append(calc_accuracy(labels, predictions))
        train_mean_acc_list.append(sum(train_acc_list) / len((train_acc_list)))
        train_acc_list = []

        for images, labels in test_loader:
            images = images.view(-1, 28 * 28)
            out = NN.forward(images)
            predictions = torch.argmax(out, 1)
            test_acc_list.append(calc_accuracy(labels, predictions))
        test_mean_acc_list.append(sum(test_acc_list) / len((test_acc_list)))
        test_acc_list = []

    # Plotting the data
    plt.plot(train_mean_acc_list, label='Train')
    plt.plot(test_mean_acc_list, label='Test')
    plt.legend()
    plt.title('Accuracy Value of train & test')
    plt.show()
    with open("q2_part1_final_model.pkl", "wb") as f: pickle.dump(NN, f)


