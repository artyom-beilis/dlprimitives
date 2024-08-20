###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
import dlprim as dp
import glob
import re
import json
import numpy as np
import argparse
import random
import time
from PIL import Image

class TinyImageNetReader(object):
    def __init__(self,basedir):
        self._dir = basedir
        self._get_labels()
        self._load_data()

    @property
    def img_size(self):
        return 3,64,64

    def _get_labels(self):
        self._labels = {}
        self._index = []
        self._names = []
        with open(self._dir + '/wnids.txt','r') as f:
            for index,lbl in enumerate(f.readlines()):
                wnid = lbl.strip()
                self._index.append(wnid)
                self._names.append(wnid)
                self._labels[wnid] = index
        r=re.compile('^(n[0-9]*)\s*(.*)$')
        with open(self._dir + '/words.txt','r') as f:
            for l in f.readlines():
                m=r.match(l.strip())
                if m:
                    wnid=m.group(1)
                    if wnid in self._labels:
                        self._names[self._labels[wnid]]=m.group(2)
        self._classes = len(self._index)
   
    @property
    def classes(self):
        return self._classes
         
    def _get_file(self,path):
        pic = Image.open(path) 
        return np.moveaxis(np.array(pic), -1, 0)

    def _load_data(self):
        self._paths={
            'train':[None for _ in range(self._classes)],
            'val':  {
                "labels":[],
                "data":None
            }
        }
        for i in range(self._classes):
            print("Loading ",self._names[i],"%3d/%d" % (i,self._classes))
            cls=self._index[i]
            search='%s/train/%s/images/*.JPEG' % (self._dir,cls)
            paths =  glob.glob(search)
            N = len(paths)
            data = np.zeros((N,3,64,64),dtype=np.uint8)
            self._paths['train'][i] = data 
            for j,p in enumerate(paths):
                data[j] = self._get_file(p)
        print("Loading test set")
        with open('%s/val/val_annotations.txt' % self._dir,'r') as f:
            rows = list(f.readlines())
            data = np.zeros((len(rows),3,64,64),dtype=np.uint8)
            self._paths['val']['data']=data
            for i,row in enumerate(rows):
                items = row.split()
                path = '%s/val/images/%s' % (self._dir,items[0])
                cls = self._labels[items[1]]
                self._paths['val']['labels'].append(cls)
                data[i] = self._get_file(path)

    @property
    def train_images(self):
        return np.sum([v.shape[0] for v in self._paths['train']])

    def fill_train_batch(self,batch,labels,random_mirror=True):
        b=batch.shape[0]
        for i in range(b):
            cls = random.randint(0,self._classes-1)
            labels[i]=cls
            img  = random.choice(self._paths['train'][cls])
            if random_mirror and random.randint(0,1) == 1:
                img = img[:,:,::-1]
            batch[i] = img.astype(np.float32) * (1/255)

    def test_data(self,batch_size):
        batch_data = np.zeros((batch_size,3,64,64),dtype=np.float32)
        batch_labels = np.zeros((batch_size,),dtype=np.float32)
        data = self._paths['val']['data']
        labels = self._paths['val']['labels']
        for indx in range(0,data.shape[0],batch_size):
            end = min(indx + batch_size,data.shape[0])
            size = end - indx
            dt = batch_data[0:size]
            lb = batch_labels[0:size]
            dt[:] = data[indx:end].astype(np.float32)*(1/255.0)
            lb[:] = labels[indx:end]
            yield dt,lb

        

def add_conv_bn_relu(n,name,inp,kern,stride,features,relu):
    n.add('Convolution2D',inputs=inp,name=name+'_cnv',outputs=name+'_cnv',options=dict(channels_out=features,
                                                            kernel=kern,
                                                            pad=(kern - 1) // 2,
                                                            stride=stride,
                                                            bias=False))
    n.add('BatchNorm',name=name+'_bn',outputs=name+'_bn')
    if relu:
        n.add('Activation',inputs=name+'_bn',outputs=name+'_bn',options=dict(activation='relu'))
    return name+'_bn'

def add_resnet_block(n,name,inp,stride,features):
    if stride > 1:
        b0 = add_conv_bn_relu(n,name+"_b0",inp,1,stride,features,relu=False)
    else:
        b0 = inp
    out = add_conv_bn_relu(n,name+"_b1a",inp,3,stride,features=features,relu=True)
    out = add_conv_bn_relu(n,name+"_b1b",out,1,stride=1,features=features,relu=False)
    n.add('Elementwise',name+'_elt',inputs=[b0,out],outputs=name+"_elt",options=dict(activation='relu'))
    return name+"_elt"
    


def make_net(batch,deploy,img_size,classes):
    n = dp.NetConfig()
    n.add_input('data',(batch,*img_size))
    n.add('Convolution2D',name='conv0',inputs='data',options=dict(channels_out=64,kernel=5,pad=2,bias=False))
    n.add('BatchNorm',name='bn0',outputs='bn0')
    n.add('Activation',name='act0',outputs='bn0',options=dict(activation='relu'))
    n.add('Pooling2D',name='pool0',outputs='pool0', options=dict(kernel=3,stride=2,pad=1))
    out  = 'pool0'
    for i,N in enumerate([128,256,512]):
        out = add_resnet_block(n,'res_%d_0' % i,out,stride=1,features=N//2)
        out = add_resnet_block(n,'res_%d_1' % i,out,stride=1,features=N//2)
        out = add_resnet_block(n,'res_%d_2' % i,out,stride=2,features=N)
    n.add('GlobalPooling',name='gp',outputs='gp',inputs=out,options=dict(mode='avg'))
    n.add('InnerProduct', outputs="ip",options=dict(outputs=classes))
    n.add('Softmax',outputs='prob')
    if not deploy:
        n.add_input('label',(batch,))
        n.add('SoftmaxWithLoss',inputs=['ip','label'],outputs='loss')
        n.set_outputs(['loss',"prob"]) # by default it gives last network output
    return n

def test(e,net,batch_size,reader):
    net.mode = dp.PREDICT
    total_loss = 0;
    total_acc = 0;
    count=0
    dtensor = net.tensor('data')
    ltensor = net.tensor('label')
    loss_tensor = net.tensor('loss')
    prob_tensor = net.tensor('prob')
    loss = np.zeros(1,dtype=np.float32)
    prob = np.zeros((batch_size,reader.classes),dtype=np.float32)
    for dt,lbl in reader.test_data(batch_size):
        n=dt.shape[0]
        if n != dtensor.shape[0]:
            dtensor.reshape(dp.Shape(*dt.shape))
            ltensor.reshape(dp.Shape(*lbl.shape))
            net.reshape()
        dtensor.to_device(dt,e)
        ltensor.to_device(lbl,e)
        net.forward(e)
        loss_tensor.to_host(loss,e)
        prob_tensor.to_host(prob[0:n],e)
        total_loss += loss[0]
        total_acc += np.sum(np.argmax(prob,axis=1)==lbl)
        count += n
    print("  Loss=%f Accuracy=%f" %(total_loss / count,total_acc / count))


def train(e,net,opt,batch,reader):
    epoch_size = (reader.train_images + batch - 1) // batch
    net.mode = dp.TRAIN
    total_loss = 0;
    total_acc = 0;
    count=0
    i=0;
    dtensor = net.tensor('data')
    ltensor = net.tensor('label')
    if dtensor.shape[0] != batch:
        dtensor.reshape(dp.Shape(batch,*reader.img_size))
        ltensor.reshape(dp.Shape(batch))
        net.reshape()
    loss_tensor = net.tensor('loss')
    prob_tensor = net.tensor('prob')
    d = np.zeros((batch,*reader.img_size),dtype=np.float32)
    l = np.zeros(batch,dtype=np.float32)
    loss = np.zeros(1,dtype=np.float32)
    prob = np.zeros((batch,reader.classes),dtype=np.float32)
    start_time = time.time()
    reader.fill_train_batch(d,l,True)
    times = np.zeros(5)
    for p in range(epoch_size):
        pt0 = time.time()
        dtensor.to_device(d,e)
        ltensor.to_device(l,e)
        pt1 = time.time()
        opt.step(net,e)
        pt2 = time.time()
        #use fact that operations are asynchronouse and prepare next batch for gpu
        if p+1 != epoch_size:
            lsave = l.copy() # save labels
            reader.fill_train_batch(d,l,True)
        pt3 = time.time()
        # back to synchronous
        loss_tensor.to_host(loss,e)
        prob_tensor.to_host(prob,e)
        total_loss += loss[0]
        total_acc += np.sum(np.argmax(prob,axis=1)==lsave) 
        count += batch
        i+=1
        total = time.time()
        if i % 100 == 0 or i+1 == epoch_size:
            time_per_item = (time.time() - start_time) * 1e3 / count * batch;
            start_time = time.time()
            print("  Iter %d: Loss=%7.6f Accuracy=%5.4f iter time %5.3f ms" %(i,total_loss / count,total_acc / count,time_per_item))
            times *=0
            total_acc = 0
            total_loss = 0
            count = 0

def main(p):
    print("Preapare data")
    imgnet = TinyImageNetReader(p.path)
    print("Prepare network")
    ctx = dp.Context(p.device)
    print(" using",ctx.name)
    e = ctx.make_execution_context()
    n = make_net(p.batch,False,imgnet.img_size,imgnet.classes)
    net = dp.Net(ctx)
    net.mode= dp.TRAIN
    net.load_from_json(n.to_str())
    with open('train.js','w') as f:
        f.write(n.to_str())
        f.write('\n')
    net.setup()
    net.initialize_parameters(e)
    ops = n.to_json()['operators']
    print("Network:")
    for op in ops:
        print("-",op['type'],json.dumps(op.get('options')))
        for inp in op['inputs']:
            print("    in: ",inp,net.tensor(inp).shape)
        for out in op['outputs']:
            print("   out: ",out,net.tensor(out).shape)
    opt = dp.Adam(ctx)
    opt.init(net,e)
    for epoch in range(p.epochs):
        print("Train Epoch:",epoch)
        train(e,net,opt,p.batch,imgnet)
        print("Valid Epoch",epoch)
        test(e,net,p.batch,imgnet)
        net.save_parameters('snap_%d.dlp' % epoch)
    make_net(p.batch,True,imgnet.img_size,imgnet.classes).save('net.json')

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--device',default='0:0')
    p.add_argument('--epochs',default=20,type=int)
    p.add_argument('--path',required=True)
    p.add_argument('--batch',default=128,type=int)
    main(p.parse_args())

