from model2 import SimpleCNN
from dl_ex5 import AnimalDataset
from torch.utils.data import DataLoader
from torch.optim import SGD
from sklearn.metrics import classification_report, accuracy_score
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
import os

def get_args():
    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--root", "-r", type= str, default= "./data", help= "Root of the dataset")
    parser.add_argument("--epochs", "-e", type= int, default= 100, help= "Number of epochs")
    parser.add_argument("--batch-size", "-b", type= int, default= 4, help= "Batch size")
    parser.add_argument("--image-size", "-i", type= int, default= 224, help= "Image size")
    parser.add_argument("--logging", "-l", type= str, default= "tensorboard")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    use_cuda = torch.cuda.is_available()
    num_workers = 0   
    num_epochs = args.epochs
    batch_size = args.batch_size
    train_dataset = AnimalDataset(root = args.root, train=True)
    train_dataLoader = DataLoader(
        dataset= train_dataset,
        batch_size= batch_size,
        shuffle=True,
        num_workers= num_workers,
        pin_memory= use_cuda,         
                drop_last=True
    )

    test_dataset = AnimalDataset(root = args.root, train=False)
    test_dataLoader = DataLoader(
        dataset= test_dataset,
        batch_size= batch_size,
        shuffle=False,
        num_workers= num_workers,
        pin_memory= use_cuda,          
                drop_last=True
    )
    writer = SummaryWriter(args.logging)
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_iters = len(train_dataLoader)
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_dataLoader)
        for iter, (images, labels) in enumerate(progress_bar):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            #forward
            optimizer.zero_grad()
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            progress_bar.set_description("epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch + 1, num_epochs, iter + 1, num_iters, loss_value.item()))
            writer.add_scalar("Train/Loss", loss_value,  epoch*num_iters+iter)

            #backward
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataLoader):
            all_labels.extend(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                prediction = model(images)
                indices = torch.argmax(prediction.cpu(), dim=1)
                all_predictions.extend(indices)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch: {}: Accuracy: {}".format(epoch + 1, accuracy))
        writer.add_scalar("Val/Accuracy", accuracy, epoch)
        
        checkpoint = {
            "epoch": epoch + 1
            ,"model": model.state_dict()
            , "optimizer": optimizer.state_dict() 
        }
        os.makedirs("save_model", exist_ok=True)
        torch.save(checkpoint, "save_model/model_lastcnn.pt")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
