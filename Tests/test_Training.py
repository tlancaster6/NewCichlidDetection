from Utils.DataLoader import BoxedImageLoader, load_boxed_annotation_data
from Utils.ConfigurationLoader import load_environment
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable


def setup_training_testing_files():
    env = load_environment()
    data = load_boxed_annotation_data(env, download=False)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = BoxedImageLoader(env, data, transform_train)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    #model.train()

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    Tensor = torch.FloatTensor
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        for batch_i, (imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor), requires_grad=True)
            targets = Variable(targets.type(Tensor), requires_grad=False)

            optimizer.zero_grad()

            loss = model(imgs, targets)
            for l in loss:
                l = l.requires_grad_(True)
                l.backward()
            optimizer.step()


            model.seen += imgs.size(0)

        if epoch % 20 == 0:
            model.save_weights("%s/%d.weights" % ('data/checkpoints/', epoch))
    print()


def main():
    setup_training_testing_files()


main()
