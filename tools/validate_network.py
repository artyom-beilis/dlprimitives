import torch
import torchvision
import json
import os
import PIL
import argparse
import time
import numpy as np
import sys

def benchmark_model(model,batch,device,warm,iters):
    model.eval()
    inp = torch.randn(batch,3,224,224).to(device)
    total_time = 0
    total_batches = 0
    total_items = 0
    print("Warming up")
    sm = torch.nn.Softmax(dim=1)
    for it in range(-warm,iters):
        start = time.time()
        res = sm(model(inp))
        torch.sum(res).item()
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

def export_model(model,batch,path,opset,ir):
    inp = torch.randn(batch,3,224,224)
    torch.onnx.export(model,inp,path,input_names = ["data"],output_names=["prob"],opset_version=opset)
    import onnx
    #from onnx import version_converter
    model = onnx.load_model(path)
    model.ir_version = ir
    onnx.save(model, path)


    



def predict_on_images(model,images,device,config):
    tw = 224
    th = 224
    mean = config['mean']
    std = config['std']
    classes = config['class_names']
    csv = []
    for i,path in enumerate(images):
        img = PIL.Image.open(path)
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
        image = torch.zeros((1,3,th,tw),dtype=torch.float32,device=device)
        for k in range(3):
            image[0,k,:,:] = torch.from_numpy(npimg[dr:dr+th,dc:dc+tw,k] * fact[k] + off[k])
        res = model(image)
        index = torch.argmax(res[0]).item()
        csv.append([path,str(index),classes[index]] + ['%8.6f' % v for v in res[0].tolist()])
    with open('report.csv','w') as f:
        for row in csv:
            line = ','.join(row) + '\n'
            f.write(line)
            sys.stdout.write(','.join(row[0:10] + ['...']) + '\n')
        



def get_config():
    base_path = os.path.dirname(os.path.abspath(__file__))
    with open(base_path + '/imagenet_predict_config.json','r') as f:
        cfg = json.load(f)
    return cfg


def main(args):
    m = getattr(torchvision.models,args.model)(pretrained = True)

    if args.export:
        export_model(m,args.batch,args.export,args.onnx_opset,args.onnx_ir)
    m.to(args.device)
    if args.benchmark:
        benchmark_model(m,args.batch,args.device,args.warm,args.iters)
    if args.images:
        predict_on_images(m,args.images,args.device,get_config())

if __name__ == '__main__': 
    p = argparse.ArgumentParser()
    p.add_argument('--model',default='vgg16')
    p.add_argument('--device',default='cuda')
    p.add_argument('--export')
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--onnx-opset',default=7)
    p.add_argument('--onnx-ir',default=3)
    p.add_argument('--batch',default=16)
    p.add_argument('--warm',default=5)
    p.add_argument('--iters',default=20)
    p.add_argument('images',nargs='*')
    r = p.parse_args()
    main(r)
