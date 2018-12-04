from .models import Res_PSP
from .models_base import resnet101_base

models ={
    'psp': Res_PSP,
    'baseline':resnet101_base,
}

def get_segmentation_model(name, **kwargs):
    return models[name.lower()](**kwargs)
