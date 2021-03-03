import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm

from autoencoder import Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, epochs, batch_size, learning_rate, data):
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    outputs = []
    for epoch in range(epochs):
        with tqdm(train_loader, unit="batch") as pbar:
            for images, labels in pbar:
                pbar.set_description(f"Epoch {epoch} Training:")
                images = images.to(device)
                reconstruction = model.forward(images)
                loss = criterion(reconstruction, images)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix_str(f"Loss: {loss.item()}")
            outputs.append((epoch, images, reconstruction))
    return outputs


def train_encoder():
    data = MNIST("MNIST", train=True, download=True, transform=transforms.ToTensor())
    model = Autoencoder(1).to(device)
    epochs = 5
    outputs = train(model, epochs, 16, 1e-3, data)
    writer = SummaryWriter("runs/autoencodermnist")
    for k in range(0, epochs):
        fig = plt.figure(figsize=(9, 2))
        images = outputs[k][1].cpu().detach().numpy()
        recon = outputs[k][2].cpu().detach().numpy()
        for i, item in enumerate(images):
            if i >= 9:
                break
            plt.subplot(2, 9, i + 1)
            plt.imshow(item[0])
        for i, item in enumerate(recon):
            if i >= 9:
                break
            plt.subplot(2, 9, 9 + i + 1)
            plt.imshow(item[0])
        writer.add_figure("Autoencoder performance", fig, global_step=k)
    model.to("cpu")
    torch.save(model.state_dict(), "autoencoderMNIST.pth")


def use_encoder():
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)
    model = Autoencoder(1)
    model.load_state_dict(torch.load("autoencoderMNIST.pth"))
    model.to(device)
    data = MNIST("MNIST", train=True, download=True, transform=transforms.ToTensor())
    writer = SummaryWriter("runs/autoencodermnist")
    while True:
        value = input("Enter two indexes to interpolate data using the trained encoder. Enter -1 to quit").split(sep=" ")
        try:
            index1, index2 = int(value[0]), int(value[1])
            if index1 == -1:
                break
            else:
                interpolate(index1, index2, data, model)
        except ValueError:
            print("Enter two integers")


def interpolate(image1, image2, model):
    x = torch.stack([image1, image2]).to(device)
    embedding = model.encoder(x)
    e1 = embedding[0]  # Embedding of first image
    e2 = embedding[1]  # Embedding of second image

    embedding_values = []
    for i in range(10):
        e = e2 * (i/10) + e1 * (10-i)/10
        embedding_values.append(e)
    embedding_values = torch.stack(embedding_values)

    recons = model.decoder(embedding_values).to("cpu").detach().numpy()
    plt.figure(figsize=(10, 2))
    # Top row are the transitions
    for i, recon in enumerate(recons):
        plt.subplot(2, 10, i + 1)
        plt.imshow(recon[0])
    # Bottom row are the original images
    plt.subplot(2, 10, 11)
    plt.imshow(image1[0])  # Get it as a 28x28
    plt.subplot(2, 10, 20)
    plt.imshow(image2[0])  # Get it as a 28x28

    return plt.gcf()

if __name__ == '__main__':
    use_encoder()
