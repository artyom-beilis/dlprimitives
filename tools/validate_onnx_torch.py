import torch
import torchvision
import json
import os
import PIL
import argparse
import time
import numpy as np
import dlprim as dp
import sys
import csv

def _prof_summary(report):
    sums=dict()
    counts=dict()
    summary=[]
    for line in [v for v in report.split('\n') if v]:
       row = [v for v in line.split(' ') if v]
       name=row[0]
       val=float(row[1])
       new_val = sums.get(name,0) + val
       new_cnt =counts.get(name,0) + 1
       sums[name ] = new_val
       counts[name] = new_cnt

    for name in sums:
        summary.append((name,sums[name],counts[name]))

    summary.sort(key = lambda x:x[1])
    print("Summary:")
    print("------")
    for r in summary:
        print("%10.5f %5d %s" % ( r[1],r[2],r[0]))
    print("------")



def export_model(model,batch,path,opset = None):
    inp = torch.randn(batch,3,224,224)
    model.eval()
    torch.onnx.export(model,inp,path,input_names = ["data"],output_names=["prob"],opset_version=opset)



class TorchModel(object):
    def __init__(self,model):
        self.model = model
    
    def eval(self,batch):
        with torch.no_grad():
            self.model.eval();
            t = torch.from_numpy(batch)
            r = self.model(t)
            return r.detach().numpy()

class DLPrimModel(object):
    def __init__(self,onnx_path,device):
        onnx_model=dp.ONNXModel();
        onnx_model.load(onnx_path)
        self.ctx = dp.Context(device)
        self.net = dp.Net(self.ctx)
        self.net.mode = dp.TRAIN
        self.net.load_model(onnx_model)

    def eval(self,batch):
        data = self.net.tensor('data')
        data.reshape(dp.Shape(*batch.shape))
        self.net.reshape()
        prob = self.net.tensor('prob')
        prob_cpu = np.zeros((prob.shape[0],prob.shape[1]),dtype=np.float32)
        q = self.ctx.make_execution_context(0)
        data.to_device(batch,q)
        self.net.forward(q)
        prob.to_host(prob_cpu,q)
        return prob_cpu




def predict_on_images(model,images,config):
    tw = 224
    th = 224
    mean = config['mean']
    std = config['std']
    classes = config['class_names']
    csv = []
    image = np.zeros((len(images),3,th,tw),dtype=np.float32)
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
        for k in range(3):
            image[i,k,:,:] = npimg[dr:dr+th,dc:dc+tw,k] * fact[k] + off[k]
    res = model.eval(image)
    for i in range(len(images)):
        index = np.argmax(res[i]).item()
        csv.append([path,str(index),classes[index]] + ['%8.6f' % v for v in res[i].tolist()])
    with open('report.csv','w') as f:
        for row in csv:
            line = ','.join(row) + '\n'
            f.write(line)
            sys.stdout.write(','.join(row[0:10] + ['...']) + '\n')
    return res


def get_config():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(base_path + '/examples/cpp/imagenet_predict_config.json','r') as f:
        cfg = json.load(f)
    return cfg


def main(args):
    m = getattr(torchvision.models,args.model)(pretrained = True)
    onnx_path = args.model + ".onnx"
    export_model(m,args.batch,onnx_path,args.onnx_opset)
    config = get_config()
    ref = predict_on_images(TorchModel(m),args.images,config)
    act = predict_on_images(DLPrimModel(onnx_path,args.device),args.images,config)
    print("Error:",np.max(np.abs(ref - act)))

if __name__ == '__main__': 
    p = argparse.ArgumentParser()
    p.add_argument('--model',default='alexnet')
    p.add_argument('--device',default='0:0')
    p.add_argument('--onnx-opset',default=9,type=int)
    p.add_argument('--batch',default=16,type=int)
    p.add_argument('images',nargs='*')
    r = p.parse_args()
    main(r)
