###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
import dlprim as dp
import mnist
import numpy as np
import argparse

def make_net(batch,deploy):
    n = dp.NetConfig()
    n.add_input('data',(batch,1,28,28))
    n.add('Convolution2D',inputs='data',options=dict(channels_out=32,kernel=5,pad=2,activation='relu'))
    n.add('Pooling2D',    options=dict(kernel=2,stride=2))
    n.add('Convolution2D',options=dict(channels_out=64,kernel=5,pad=2,activation='relu'))
    n.add('Pooling2D',    options=dict(kernel=2,stride=2))
    n.add('InnerProduct', options=dict(outputs=256,activation='relu'))
    n.add('InnerProduct', outputs="ip2",options=dict(outputs=10))
    n.add('Softmax',outputs='prob')
    if not deploy:
        n.add_input('label',(batch,))
        n.add('SoftmaxWithLoss',inputs=['ip2','label'],outputs='loss')
        n.set_outputs(['loss',"prob"]) # by default it gives last network output
    return n

def train_data():
    data  = mnist.train_images().reshape(-1,1,28,28).astype(np.float32)*(1/255.0)
    label = mnist.train_labels().astype(np.float32)
    return data,label

def test_data():
    data  = mnist.test_images().reshape(-1,1,28,28).astype(np.float32)*(1/255.0)
    label = mnist.test_labels().astype(np.float32)
    return data,label


def test(e,net,batch,data,labels):
    net.mode = dp.PREDICT
    N=data.shape[0]
    total_loss = 0;
    total_acc = 0;
    count=0
    for p in range(0,N,batch):
        d = data[p:p+batch]
        l = labels[p:p+batch]
        dtensor = net.tensor('data')
        ltensor = net.tensor('label')
        loss_tensor = net.tensor('loss')
        prob_tensor = net.tensor('prob')
        n=d.shape[0]
        if n != dtensor.shape[0]:
            dtensor.reshape(dp.Shape(*d.shape))
            ltensor.reshape(dp.Shape(*l.shape))
            net.reshape()
        dtensor.to_device(d,e)
        ltensor.to_device(l,e)
        net.forward(e)
        loss = np.zeros(1,dtype=np.float32)
        prob = np.zeros((n,10),dtype=np.float32)
        loss_tensor.to_host(loss,e)
        prob_tensor.to_host(prob,e)
        total_loss += loss[0]
        total_acc += np.sum(np.argmax(prob,axis=1)==l)
        a = np.argmax(prob,axis=1)
        b = l
        count += n
    print("Test: Loss=%f Accuracy=%f" %(total_loss / count,total_acc / count))


def train(e,net,opt,batch,data,labels):
    net.mode = dp.TRAIN
    N=data.shape[0]
    total_loss = 0;
    total_acc = 0;
    count=0
    i=0;
    for p in range(0,N,batch):
        d = data[p:p+batch]
        l = labels[p:p+batch]
        dtensor = net.tensor('data')
        ltensor = net.tensor('label')
        loss_tensor = net.tensor('loss')
        prob_tensor = net.tensor('prob')
        n=d.shape[0]
        if n != dtensor.shape[0]:
            dtensor.reshape(dp.Shape(*d.shape))
            ltensor.reshape(dp.Shape(*l.shape))
            net.reshape()
        dtensor.to_device(d,e)
        ltensor.to_device(l,e)
        opt.step(net,e)
        loss = np.zeros(1,dtype=np.float32)
        prob = np.zeros((n,10),dtype=np.float32)
        loss_tensor.to_host(loss,e)
        prob_tensor.to_host(prob,e)
        total_loss += loss[0]
        total_acc += np.sum(np.argmax(prob,axis=1)==l)
        a = np.argmax(prob,axis=1)
        b = l
        count += n
        i+=1
        if i % 100 == 0 or p+batch >= N:
            print("  %6d: Loss=%f Accuracy=%f" %(i,total_loss / count,total_acc / count))
            total_acc = 0
            total_loss = 0
            count = 0


def main(p):
    print("Preapare data")
    tr_d,tr_l = train_data()
    ts_d,ts_l = test_data()
    print("Prepare network")
    ctx = dp.Context(p.device)
    print(" using",ctx.name)
    e = ctx.make_execution_context()
    n = make_net(p.batch,False)
    net = dp.Net(ctx)
    net.mode= dp.TRAIN
    net.load_from_json(n.to_str())
    net.setup()
    net.initialize_parameters(e)
    opt = dp.Adam(ctx)
    opt.init(net,e)
    for epoch in range(5):
        print("Epoch",epoch)
        train(e,net,opt,p.batch,tr_d,tr_l)
        print("Testing",epoch)
        test(e,net,p.batch,tr_d,tr_l)
    net.save_parameters('mnsit.dlp')
    make_net(p.batch,True).save('mnist.json')

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--device',default='0:0')
    p.add_argument('--batch',default=128,type=int)
    main(p.parse_args())

