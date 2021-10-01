import torchvision.models.detection

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=3, pretrained_backbone=True)
model = torchvision.models.detection.ssd300_vgg16(pretrained=False)
model.train()