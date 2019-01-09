from .CE2P import Res_PSP
from .models_base import resnet101_base
from .norm_base import norm_resnet101
#from .depth2norm import depth2norm
#from .basedorn import basedorn
from .deeplab_resnet import Res_Deeplab
from .dorn import DORN
models ={
    'psp': Res_PSP,
    'baseline':resnet101_base,
    'normbase':norm_resnet101,
 #   'depth2norm':depth2norm,
 #   'basedorn':basedorn,
    'deeplabdorn':Res_Deeplab,
    'dorn':DORN
}

def get_segmentation_model(name, **kwargs):
    return models[name.lower()](**kwargs)
