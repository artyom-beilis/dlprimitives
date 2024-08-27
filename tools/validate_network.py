###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
import torch

import torchvision
import json
import os
import PIL
import argparse
import time
import numpy as np
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


def benchmark_model(model,batch,device,warm,iters,train,use_solver,profile):

    def _sync():
        if device.find('opencl')==0 or device.find('privateuseone')==0 or device.find('ocl')==0:
            torch.ocl.synchronize()
        elif device.find('xpu')==0:
            torch.xpu.synchronize()
        elif device.find('cuda')==0:
            torch.cuda.synchronize()

    if train:
        model.train()
    else:
        use_solver = False
        model.eval()
    #inp_cpu = torch.randn(batch,3,224,224)
    shape = (batch,3,224,224)
    inp_cpu = torch.empty(shape,dtype=torch.float32)
    torch.randn(shape,out=inp_cpu)
    total_time = 0
    total_io = 0
    total_fw = 0
    total_bw = 0
    total_zero = 0
    total_update = 0
    total_batches = 0
    total_items = 0
    print("Warming up")
    if train:
        sm = torch.nn.LogSoftmax(dim=1)
        nll = torch.nn.NLLLoss()
        lbl_cpu = torch.randint(1000,size=(batch,))
    if use_solver:
        optimizer = torch.optim.Adam(model.parameters())
    for it in range(-warm,iters):
        def run_step():
            start = time.time()
            if use_solver:
                optimizer.zero_grad()
                _sync()
                zero_point = time.time()
            else:
                zero_point = start

            inp = inp_cpu.to(device)
            if train:
                lbl = lbl_cpu.to(device)

            _sync()
            io_point = time.time()
            res = model(inp)
            if train:
                res = sm(res)
                l=nll(res,lbl)
                _sync()
                fwd_end = time.time()
                l.backward()
                _sync()
                bwd_end = time.time();
                if use_solver:
                    optimizer.step()
                    _sync()
                    solver_end = time.time()
                else:
                    solver_end = bwd_end
            else:
                res.to('cpu') 
                _sync()
                fwd_end = time.time()
                solver_end = fwd_end
                bwd_end = fwd_end
            end = time.time()
            return start,end,zero_point,io_point,fwd_end,bwd_end,solver_end
        if it == 0 and profile:
            with torch.ocl.profile(device,"prof.csv"):
                start,end,zero_point,io_point,fwd_end,bwd_end,solver_end=run_step()
        else:
            start,end,zero_point,io_point,fwd_end,bwd_end,solver_end = run_step()
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
            if train:
                total_fw += fwd_end - start
                total_bw += end - fwd_end
                total_io += io_point - zero_point
                total_zero += zero_point - start
                total_update += solver_end - bwd_end
    print("Time per item  %1.3f ms" %(total_time / total_items *1e3))
    if train:
        print("Time fwd batch  %1.3f ms" %(total_fw / total_batches *1e3))
        print("Time bwd batch  %1.3f ms" %(total_bw / total_batches *1e3))
        print("Time io  batch  %1.3f ms" %(total_io / total_batches *1e3))
        print("Time zro batch  %1.3f ms" %(total_zero / total_batches *1e3))
        print("Time opt batch  %1.3f ms" %(total_update  / total_batches *1e3))

    print("Time per batch %1.3f ms" %(total_time / total_batches *1e3))

def export_model(model,batch,path,opset,ir,train):
    inp = torch.randn(batch,3,224,224)
    model.eval()
    if train:
        extra =dict( training=torch.onnx.TrainingMode.TRAINING,do_constant_folding=False)
    else:
        extra = dict(do_constant_folding=True)
    torch.onnx.export(model,inp,path,input_names = ["data"],output_names=["prob"],opset_version=opset,**extra)
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
    model.eval()
    image = torch.zeros((len(images),3,th,tw),dtype=torch.float32)
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
            image[i,k,:,:] = torch.from_numpy(npimg[dr:dr+th,dc:dc+tw,k] * fact[k] + off[k])
    image = image.to(device)
    res = model(image)
    for i in range(len(images)):
        index = torch.argmax(res[i]).item()
        csv.append([path,str(index),classes[index]] + ['%8.6f' % v for v in res[i].tolist()])
    with open('report.csv','w') as f:
        for row in csv:
            line = ','.join(row) + '\n'
            f.write(line)
            sys.stdout.write(','.join(row[0:10] + ['...']) + '\n')
        



def get_config():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(base_path + '/examples/cpp/imagenet_predict_config.json','r') as f:
        cfg = json.load(f)
    return cfg


def main(args):
    m = getattr(torchvision.models,args.model)(weights = 'DEFAULT')
    #print("Mean",m.bn1.running_mean.tolist()[:4])
    #print("Var",m.bn1.running_var.tolist()[:4])
    #print("W",m.bn1.weight.tolist()[:4])
    #print("B",m.bn1.bias.tolist()[:4])
    if args.export:
        export_model(m,args.batch,args.export,args.onnx_opset,args.onnx_ir,args.train)
    m.to(args.device)
    if args.images:
        with torch.no_grad():
            predict_on_images(m,args.images,args.device,get_config())
    if args.benchmark:
        if args.train:
            benchmark_model(m,args.batch,args.device,args.warm,args.iters,args.train,args.solver,args.profile)
        else:
            with torch.no_grad():
                benchmark_model(m,args.batch,args.device,args.warm,args.iters,args.train,False,args.profile)

if __name__ == '__main__': 
    p = argparse.ArgumentParser()
    p.add_argument('--model',default='vgg16')
    p.add_argument('--device',default='cuda')
    p.add_argument('--export')
    p.add_argument('--solver',action='store_true')
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--train',action='store_true')
    p.add_argument('--profile',action='store_true',default=False)
    p.add_argument('--onnx-opset',default=9,type=int)
    p.add_argument('--onnx-ir',default=3,type=int)
    p.add_argument('--batch',default=16,type=int)
    p.add_argument('--warm',default=5,type=int)
    p.add_argument('--iters',default=20,type=int)
    p.add_argument('images',nargs='*')
    r = p.parse_args()
    if r.device.find('ocl')==0 or r.device.find('privateuseone')==0:
        import pytorch_ocl
        if r.profile:
            torch.ocl.enable_profiling(r.device)
    if r.device.find('xpu')==0:
        import intel_extension_for_pytorch
    main(r)
