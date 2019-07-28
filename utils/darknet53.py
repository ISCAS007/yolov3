# -*- coding: utf-8 -*-
import torch
import os
import sys
if '.' not in sys.path:
    sys.path.insert(0,'.')
from models import Darknet,load_vgg_weights

def darknet53(cfg,img_size=(416, 416)):
    return Darknet(cfg,img_size)

if __name__ == '__main__':
    net=darknet53('cfg/yolov3-vgg16.cfg',(416,416))
#    print(net)

    cutoff=load_vgg_weights(net,'vgg16',18)
    net.train()

    img=torch.rand(2,3,416,416)

    output=net.forward(img)

    for o in output:
        print(o.shape,torch.max(o))