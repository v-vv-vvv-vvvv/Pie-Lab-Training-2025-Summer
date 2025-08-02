from dataset import load_dataset
from dataset import cifar_dataset
from torch.utils.data import Dataset, DataLoader

from model import UNet
from model import Diffusion
import numpy

import torch
from tqdm import tqdm
from visualize import gen_images

def main():
    dataDir = 'cifar/cifar-10'
    epochs = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_data, train_labels, test_data, test_labels = load_dataset(dataDir)
    all_data = numpy.concatenate([train_data,test_data],axis=0)
    all_labels = train_labels + test_labels
    train_loader = DataLoader(cifar_dataset(all_data, all_labels), batch_size=256, shuffle=True)
    # test_loader = DataLoader(cifar_dataset(test_data, test_labels))

    model = UNet().to(device)
    diff = Diffusion(model=model, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    loos_all = []

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            loss = diff.training_losses(images, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch}: loss = {loss.item():.4f}")
        loos_all.append(loss.item())

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        #'best_acc': best_acc
    }, 'checkpoint-ddim-30.pth')

    print(loos_all)
    print("OMG!")
    # gen_images(diff, model)

if __name__=="__main__":
    main()