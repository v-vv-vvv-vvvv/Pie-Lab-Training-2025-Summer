from model import UNet
from model import Diffusion
import numpy

import torch
from visualize import gen_images

def evaluate():
    dataDir = 'cifar/cifar-10'
    checkpoint_path = 'checkpoint-diff-20.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # train_data, train_labels, test_data, test_labels = load_dataset(dataDir)
    # all_data = numpy.concatenate([train_data,test_data],axis=0)
    # all_labels = train_labels + test_labels
    # train_loader = DataLoader(cifar_dataset(all_data, all_labels), batch_size=256, shuffle=True)

    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'epoch': epoch,
    #     #'best_acc': best_acc
    # }, 'checkpoint-diff-20.pth')

    model = UNet().to(device)
    diff = Diffusion(model=model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    gen_images(diff, model, save_dir='out_ddim')

if __name__=="__main__":
    evaluate()