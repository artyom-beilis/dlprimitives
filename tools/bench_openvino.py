###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
import os
import json
import argparse
import time
import sys
import numpy as np
from PIL import Image
from openvino.inference_engine import IECore


def predict_on_images(model,images,config):
    tw = 224
    th = 224
    mean = config['mean']
    std = config['std']
    classes = config['class_names']
    csv = []
    for i,path in enumerate(images):
        img = Image.open(path)
        npimg = np.array(img).astype(np.float32) * (1.0 / 255)
        h = npimg.shape[0]
        w = npimg.shape[1]
        assert h>=th
        assert w>=tw
        assert npimg.shape[2] == 3
        fact = 1.0 / np.array(std)
        off  = -np.array(mean) * fact
        dr = (h - th) // 2
        dc = (w - tw) // 2
        image = np.zeros((1,3,th,tw),dtype=np.float32)
        for k in range(3):
            image[0,k,:,:] = npimg[dr:dr+th,dc:dc+tw,k] * fact[k] + off[k]
        res = model(image)
        index = np.argmax(res[0]).item()
        csv.append([path,str(index),classes[index]] + ['%8.6f' % v for v in res[0].tolist()])
    with open('report.csv','w') as f:
        for row in csv:
            line = ','.join(row) + '\n'
            f.write(line)
            sys.stdout.write(','.join(row[0:10] + ['...']) + '\n')
        

def benchmark_model(model,batch,warm,iters):
    inp = np.random.random((batch,3,224,224)).astype(np.float32)
    total_time = 0
    total_batches = 0
    total_items = 0
    print("Warming up")
    for it in range(-warm,iters):
        start = time.time()
        res = model(inp)
        end = time.time()
        msg = ''
        if it == -warm:
            msg = 'warming up'
        elif it == 0:
            msg = 'started'
        print("Step %2d %5.3fms  %s" % (it, (end-start) * 1e3,msg))
        if it>=0:
            total_time += end-start
            total_items += batch
            total_batches += 1
    print("Time per item  %1.3fms" %(total_time / total_items *1e3))
    print("Time per batch %1.3fms" %(total_time / total_batches *1e3))


def get_config():
    base_path = os.path.dirname(os.path.abspath(__file__))
    with open(base_path + '/imagenet_predict_config.json','r') as f:
        cfg = json.load(f)
    return cfg


class VinoModel(object):
    def __init__(self,path,device):
        self.ie = IECore()
        self.net = self.ie.read_network(model=path)
        assert len(self.net.input_info) == 1 and len(self.net.outputs) == 1
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))
        self.net.input_info[self.input_blob].precision = 'FP32'
        self.net.outputs[self.out_blob].precision = 'FP32'
        assert self.net.outputs[self.out_blob].shape[1] == 1000
        self.exec_net = self.ie.load_network(network = self.net,device_name = device)

    def __call__(self,images):
        return self.exec_net.infer(inputs = {self.input_blob:images})['prob']


def get_model(path,dev):
    model = VinoModel(path,dev)
    return model


def main(args):
    m=get_model(args.model,args.device)

    if args.benchmark:
        benchmark_model(m,args.batch,args.warm,args.iters)
    if args.images:
        predict_on_images(m,args.images,get_config())

if __name__ == '__main__': 
    p = argparse.ArgumentParser()
    p.add_argument('--model',default='vgg16')
    p.add_argument('--device',default='GPU')
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--batch',default=16,type=int)
    p.add_argument('--warm',default=5,type=int)
    p.add_argument('--iters',default=20,type=int)
    p.add_argument('images',nargs='*')
    r = p.parse_args()
    main(r)
