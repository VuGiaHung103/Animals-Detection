from model2 import SimpleNeuralNetwork
from dl_ex5 import AnimalDataset
from torch.utils.data import DataLoader
from torch.optim import SGD
from sklearn.metrics import classification_report
import torch.nn as nn
import torch

if __name__ == '__main__':
    num_epochs = 100
    train_dataset = AnimalDataset(root = "./animals", train=True)
    train_dataLoader = DataLoader(
        dataset= train_dataset,
        batch_size= 64,
        shuffle=True,
        num_workers= 4,
        drop_last=True
    )

    test_dataset = AnimalDataset(root = "./animals", train=False)
    test_dataLoader = DataLoader(
        dataset= test_dataset,
        batch_size= 128,
        shuffle=True,
        num_workers= 0,
        pin_memory= True,
        drop_last=True
    )
    model = SimpleNeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_iters = len(train_dataLoader)
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(num_epochs):
        model.train()
        for iter, (images, labels) in enumerate(train_dataLoader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            #forward
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            # print("epoch {}/{}. Iteration {}/{}. Loss {}".format(epoch + 1, num_epochs, iter + 1, num_iters, loss_value))

            #backward
            optimizer.zero_grad()
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
        print("Epoch: {}".format(epoch + 1))
        print(classification_report(all_labels, all_predictions, zero_division=0))
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_path = "./checkpoints/animal_model_best.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Save weights success at: {checkpoint_path}")