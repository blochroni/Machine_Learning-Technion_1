import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import pickle
from hw1_313606816_train import Neural_Network


def accuracy(loader, NN):
    summary = 0
    correct = 0
    for images, labels in loader:
        images = images.view(-1, 28 * 28)
        out = NN.forward(images)
        pred = torch.argmax(out, 1)
        summary += labels.size(0)
        correct += (pred == labels).sum()
    return correct / summary


def evaluate_hw1():

    # Hyper Parameters
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


    # getting the trained model
    NN = pickle.load(open("q2_part1_final_model.pkl", 'rb'))


    print(f"Test Accuracy: {accuracy(test_loader, NN)}, Train Accuracy: {accuracy(train_loader, NN)}")


if __name__ == '__main__':
    evaluate_hw1()




