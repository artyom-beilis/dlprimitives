###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
import json
import os
from PIL import Image
import argparse
import time
import numpy as np
import sys
import csv

layer = None

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



def export_torch_model(model,batch,path,opset = None):
    import torch
    inp = torch.randn(batch,3,224,224)
    model.eval()
    torch.onnx.export(model,inp,path,input_names = ["data"],output_names=["prob"],opset_version=opset)

class CaffeModel(object):
    def __init__(self,proto,params,device = -1):
        import caffe
        if device >= 0:
            caffe.set_mode_gpu()
            caffe.set_device(device)
        self.net = caffe.Net(proto,caffe.TEST)
        self.net.copy_from(params)
    
    def eval(self,batch):
        global layer
        data = self.net.blobs[self.net.inputs[0]]
        lname = layer if layer else self.net.outputs[0]
        prob = self.net.blobs[lname] 
        if data.data.shape[0] != batch.shape[0]:
            data.reshape(*batch.shape)
            self.net.reshape()
        np.copyto(data.data,batch)
        host_prob = np.zeros(prob.data.shape,dtype=np.float32)
        self.net.forward()
        np.copyto(host_prob,prob.data)
        return host_prob


class DLPrimModel(object):
    def __init__(self,proto,params,device):
        import dlprim as dp
        caffe_model=dp.CaffeModel();
        caffe_model.load(proto,params)
        self.ctx = dp.Context(device)
        self.net = dp.Net(self.ctx)
        #self.net.keep_intermediate_tensors = True
        self.net.mode = dp.PREDICT
        base_path = proto.replace('.prototxt','')
        with open(base_path +'.json','w') as f:
            f.write(caffe_model.network)
            f.write('\n')
        self.net.load_model(caffe_model)
        self.net.save_parameters(base_path + '.dlp')

    def eval(self,batch):
        import dlprim as dp
        data = self.net.tensor(self.net.input_names[0])
        if data.shape[0] != batch.shape[0]:
            data.reshape(dp.Shape(*batch.shape))
            self.net.reshape()
        global layer
        lname = layer if layer else self.net.output_names[0]
        prob = self.net.tensor(lname)
        prob_cpu = np.zeros(prob.shape,dtype=np.float32)
        q = self.ctx.make_execution_context(0)
        data.to_device(batch,q)
        self.net.forward(q)
        prob.to_host(prob_cpu,q)
        q.finish()
        return prob_cpu


def predict_on_images(model,images,config):
    tw = 224
    th = 224
    mean = config['mean']
    mean = np.array(mean) * 255;
    classes = config['class_names']
    csv = []
    image = np.zeros((len(images),3,th,tw),dtype=np.float32)
    for i,path in enumerate(images):
        img = Image.open(path)
        npimg = np.array(img).astype(np.float32)
        h = npimg.shape[0]
        w = npimg.shape[1]
        assert h>=th
        assert w>=tw
        assert npimg.shape[2] == 3
        dr = (h - th) // 2
        dc = (w - tw) // 2
        # RGB 2 BGR
        for k in range(3):
            image[i,2-k,:,:] = npimg[dr:dr+th,dc:dc+tw,k] - mean[k]
    res = model.eval(image)
    for i in range(len(images)):
        index = np.argmax(res[i]).item()
        csv.append([images[i],str(index),classes[index]] + ['%8.6f' % v for v in res[i].tolist()])
    with open('report.csv','w') as f:
        for row in csv:
            line = ','.join(row) + '\n'
            f.write(line)
            sys.stdout.write(','.join(row[0:10] + ['...']) + '\n')
    return res

def predict_on_data(model,data):
    data = np.load(data)
    return model.eval(data)


def get_config():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(base_path + '/examples/cpp/imagenet_predict_config.json','r') as f:
        cfg = json.load(f)
    return cfg

def run_bm(name,model,bs,data = None):
    if data is None:
        data = np.random.randn(bs,3,224,224).astype(np.float32)
    else:
        data = np.load(data)
    vals = []
    import time
    print("Bechmark ",name)
    for i in range(-3,10):
        start = time.time()
        model.eval(data)
        end = time.time()
        print("%3d %5.3f ms" % (i,(end-start)*1000))
        if i>=0:
            vals.append(end-start)
    print(name,"Mean %5.3fms" % (1000*np.mean(vals)))


def main(args):
    proto = args.proto
    param = args.param
    config = get_config()
    src_model = CaffeModel(proto,param,0)
    global layer
    layer = None
    if args.data:
        ref = predict_on_data(src_model,args.data)
    else:
        ref = predict_on_images(src_model,args.images,config)
    if args.benchmark:
        run_bm('Caffe',src_model,len(args.images),args.data)
    src_model = None
    print("DLPrimitives")
    m = DLPrimModel(proto,param,args.device)
    if args.data:
        act = predict_on_data(m,args.data)
    else:
        act = predict_on_images(m,args.images,config)
    if args.benchmark:
        run_bm('dlprimitives',m,len(args.images),args.data)
    err = np.max(np.abs(ref - act))
    del m
    print("Error:",err)
    if err > 1e-4:
        print(ref)
        print(act)
        print("Failed")
        sys.exit(1)

if __name__ == '__main__': 
    p = argparse.ArgumentParser()
    p.add_argument('--proto')
    p.add_argument('--param')
    p.add_argument('--device',default='0:0')
    p.add_argument('--batch',default=4,type=int)
    p.add_argument('--data')
    p.add_argument('--benchmark',default=False,action='store_true')
    p.add_argument('images',nargs='*')
    r = p.parse_args()
    r.batch = len(r.images)
    main(r)
