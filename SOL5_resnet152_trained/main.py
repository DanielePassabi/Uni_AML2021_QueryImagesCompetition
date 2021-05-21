import sys
from data_loader import load_stl,load_stl_bin
from model import ResNet
from solver import ImageClassifier

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



def main():
    # way 1: outside of pytorch
    # we init the model
    model = ResNet()

    # we load data
    training_data, training_labels, test_data, test_labels = load_stl()

    # we init IC
    ic = ImageClassifier(batch_size=32, epochs=100)

    # train and eval
    ic.train(training_data, training_labels, test_data, test_labels, model)

def main_pytorch():
    # way 2, pytorch uses parallel multi procesing dataloading -> Dataloader

    # init the model
    model = ResNet()

    # we define data transformations
    data_transforms = {'train': transforms.Compose([
                       transforms.RandomResizedCrop(224),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       'val': transforms.Compose([
                       transforms.Resize(256),
                       transforms.CenterCrop(224),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # we init datasets
    train_dataset = datasets.STL10(root='data', download=False,
                                   split='train', transform=data_transforms['train'])
    test_dataset = datasets.STL10(root='data', download=False,
                                  split='test', transform=data_transforms['val'])

    # we define dataloaders -> they use multiprocessing for loading data
    train_loader = DataLoader(train_dataset, batch_size=32,
                              num_workers=2, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=32,
                             num_workers=2,shuffle=False)
    # init IC
    ic = ImageClassifier(batch_size=32, epochs=100)

    # train and eval with pytorch classes
    ic.train_pytorch(train_loader, test_loader, model)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
