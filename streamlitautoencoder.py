import streamlit as st
import torch
import torchvision.transforms as transforms
from continuum.datasets import MNIST, FashionMNIST
from continuum.scenarios import ClassIncremental

from autoencoder import Autoencoder
from train_autoencoder import interpolate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache(allow_output_mutation=True)
def load_dataset(dataset):
    if dataset == "MNIST":
        data = MNIST("MNIST", train=True, download=True)
        data = ClassIncremental(data, nb_tasks=10, transformations=transformations)
    elif dataset == "FashionMNIST":
        data = FashionMNIST("FashionMNIST", train=True, download=True)
        data = ClassIncremental(data, nb_tasks=10, transformations=transformations)
    return data

@st.cache
def load_model(dataset):
    model = Autoencoder(1)
    if dataset == "MNIST":
        model.load_state_dict(torch.load("autoencoderMNIST.pth"))
    elif dataset == "FashionMNIST":
        model.load_state_dict(torch.load("autoencoderFashionMNIST.pth"))
    model.to(device)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Visualizing autoencoder embeddings")

dataset = st.selectbox("Choose dataset", options=["MNIST", "FashionMNIST"])
transformations = [transforms.ToTensor()]

loading = st.text(f"Loading {dataset} dataset")

data = load_dataset(dataset)

loading.text("Loading the autoencoder")

model = load_model(dataset)

loading.text("Ready")


def interactive():
    if dataset == "MNIST":
        st.write("Enter two integers.")
        index1 = st.number_input("Index 1", value=1, format="%d", min_value=0, max_value=9)
        index2 = st.number_input("Index 2", value=9, format="%d", min_value=0, max_value=9)

        image1 = data[index1].get_random_samples(1)[0][0]   # This returns a tuple of images,labels, tasks. The first
        image2 = data[index2].get_random_samples(1)[0][0]   # [0] selects the images, the second [0] selects the image
    elif dataset == "FashionMNIST":
        st.write("Select two classes.")
        classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        choice1 = st.selectbox("Choose class for first image", options=classes)
        choice2 = st.selectbox("Choose class for second image", options=classes)

        index1 = classes.index(choice1)
        index2 = classes.index(choice2)

        image1 = data[index1].get_random_samples(1)[0][0]
        image2 = data[index2].get_random_samples(1)[0][0]

    fig = interpolate(image1, image2, model)
    st.pyplot(fig)

# Really nasty trick to force a reroll of the images. At the moment, im not sure how this could be done in a nicer way
if not st.button("Hit me to reroll images"):
    interactive()
else:
    interactive()




