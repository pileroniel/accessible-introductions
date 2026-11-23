"""
This module builds the CNN model and the evaluation transform, which is shared by the training and the app,
the same way our features.py can we used

I keep the model and it's preprocessing here, not in the training script, so that 
the train_cnn module and my app woul build the exact same model and prepare images the 
exact way it did. This is shared  the same way as features.py hog baseline is also shared

The tutorials I used include

- Transfer learning tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- torchvision ResNet-50:      https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
- Transforms:                 https://pytorch.org/vision/stable/transforms.html
"""

from pathlib import Path

import torch 
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.models import ResNet50_Weights, resnet50

# saving these configs inside the checkpoint so my app rebuilds the model and 
# the preprocessing exactly

CNN_PARAMS = {
    "arch":"resnet50",
    "img_size":224, # image size resnet-50 was trained with
    "pad_to_square":True, # pad the image instead of squashing so a long dress still looks long
    "dropout":0.2, # dropout in front of the new CNN head, a little extra reguralization
}

# the backbone learned on images normalised with these imagenet statistics, so mine 
# must also match to obtain the same results

# got this from this article 
# https://medium.com/@engr.akhtar.awan/how-to-fine-tune-the-resnet-50-model-on-your-target-dataset-using-pytorch-187abdb9beeb 

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PAD_FILL = tuple(int(m * 255) for m in IMAGENET_MEAN)  # mean grey, which becomes about 0 after normalising

# which weights train in phase 2, by parameter prefix

UNFREEZE = {
    "layer4": ("layer4", "fc"),
    "layer3": ("layer3", "layer4", "fc"),
    "all": ("",),
}

def pad_to_square(img: Image.Image) -> Image.Image:
    """I centre the image on a square canvas instead of stretching it into a square.
    This keeps the garment's shape, the same way we used the aspect ratio did in our baseline resizing mode, i.e at build_garment_subset.py, line 75"""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), PAD_FILL)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2)) # center the image by dividing the margin to be equal on each side
    return canvas


def make_transform(train: bool, p=CNN_PARAMS):
    """I return the preprocessing for training (with random augmentation) or for val, test and my app (fixed)."""
    size = p["img_size"]
    steps = [T.Lambda(pad_to_square)] if p["pad_to_square"] else []
    if train:
        steps += [
            # I do small and random crops so hems and collars survive; ratio near 1 so I don't re-stretch the shape
            T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),                     # a mirrored jacket is still a jacket
            T.ColorJitter(brightness=0.2, contrast=0.2),  # no hue jitter, colour is a later attribute
        ]
    else:
        steps += [T.Resize((size, size))]
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)

# TODO-  understand how the linear model replaces the head, and what dropping the head out means

def build_model(num_classes: int, p=CNN_PARAMS, pretrained: bool = True) -> nn.Module:
    """I load ResNet-50 and swap its 1000-class ImageNet head for a new head with one output per garment class."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    ## INFO torch.nn.Sequential-  A sequential container.Modules will be added to it in the order they are passed in the constructor.
    ## https://docs.pytorch.org/docs/2.15/generated/torch.nn.Sequential.html 

    model.fc = nn.Sequential(nn.Dropout(p["dropout"]), nn.Linear(model.fc.in_features, num_classes))
    return model


def set_trainable(model: nn.Module, phase: int, unfreeze: str = "layer3") -> int:
    """I choose which layers learn, and return how many weights there are.
    1. Phase 1: only my new head. The backbone is a frozen therefore acts like a feature extractor of 2048 features.
    2. Phase 2: my head plus the top of the backbone (which is mostly a layer that discovers shapes and the whole thing). This layer therefore adapt to garments."""
    prefixes = ("fc",) if phase == 1 else UNFREEZE[unfreeze]
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(prefixes) 
        # this sets the model to do gradient descent so as to learn the weights
    # return the number or elements/weights in the parameters that use gradient descent
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_batchnorm_stats(model: nn.Module):
    """BatchNorm layers keep running averages that still update  in train mode, even when their weights are frozen.
    I switch every frozen BatchNorm layer to eval mode so a frozen layer really stays the same.
    I call this right after model.train() at the start of every epoch."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def save_checkpoint(path, model, classes, extra=None, p=CNN_PARAMS):
    """I save the weights together with the class names and preprocessing settings, so my app can reload them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": list(classes), "cnn_params": dict(p),
                **(extra or {})}, path)


def load_checkpoint(path, device=None):
    """I rebuild my trained model for inference. Returns (model in eval mode, class names, eval transform).
    With no device given, I use the GPU when there is one and the CPU otherwise."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device, weights_only=True)
    p = ckpt["cnn_params"]
    model = build_model(len(ckpt["classes"]), p, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["classes"], make_transform(train=False, p=p)


@torch.no_grad()
def predict_image(model, img: Image.Image, transform):
    """I return class probabilities (summing to 1) for one PIL image. My fallback rule (Algorithm 1) uses these.
    I send the image to whichever device my model is on, so the two always match."""
    device = next(model.parameters()).device
    x = transform(img.convert("RGB")).unsqueeze(0).to(device)
    return torch.softmax(model(x), dim=1)[0].cpu().numpy()
