###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import json


class Net2(nn.Module):
    def __init__(self,batch):
        super(Net2, self).__init__()
        self.cnv1 = nn.Conv2d(1, 16,3,padding=1)
        self.cnv2 = nn.Conv2d(16,24,3,padding=1)
        self.cnv2a = nn.Conv2d(16,8,1,padding=0,bias=False)
        self.bn2a = nn.BatchNorm2d(8)
        self.cnv2b = nn.Conv2d(8,24,3,padding=1)
        self.fc = nn.Linear(24, 10)
        
        self.p1   = nn.MaxPool2d(2,stride=2)
        self.p2   = nn.AdaptiveAvgPool2d((1,1))

        self.mjs = dict(
            inputs = [dict(shape=[batch,1,28,28],name='data'),
                      dict(shape=[batch,1],name='label')],
            outputs= ["loss"],
            operators = [
                dict(name="cnv1",
                     type="Convolution2D",
                     inputs=["data"],
                     outputs=["cnv1"],
                     options = dict(
                        channels_out = 16,
                        kernel=list(self.cnv1.kernel_size),
                        pad=list(self.cnv1.padding),
                        activation="relu"
                    )
                ),
                dict(name="p1",
                     type="Pooling2D",
                     inputs=["cnv1"],outputs=["p1"],
                     options = dict(kernel=2,stride=2)),
                dict(name="cnv2",
                     type="Convolution2D",
                     inputs=["p1"],
                     outputs=["cnv2"],
                     options = dict(
                        channels_out = 24,
                        kernel=list(self.cnv2.kernel_size),
                        pad=list(self.cnv2.padding)
                    )
                ),
                dict(name="cnv2a",
                     type="Convolution2D",
                     inputs=["p1"],
                     outputs=["cnv2a"],
                     options = dict(channels_out=8,kernel = 1,pad = 0,bias=False)),
                dict(name="bn2a",
                     type="BatchNorm",
                     inputs=["cnv2a"],
                     outputs=["bn2a"],
                     options=dict(features=8)),
                dict(name="relu_bn2",
                     type="Activation",
                     inputs=["bn2a"],outputs=["bn2a"],
                     options=dict(activation='relu')),
                dict(name="cnv2b",
                     type="Convolution2D",
                     inputs=["bn2a"],
                     outputs=["cnv2b"],
                     options = dict(channels_out=24,kernel = 3,pad = 1)),
                dict(name="elt",
                     type="Elementwise",
                     inputs=["cnv2","cnv2b"],
                     outputs=["elt"],
                     options= dict(operations="sum")),
                dict(name="elt_relu",type="Activation",
                     inputs=["elt"],outputs=["elt"],
                     options=dict(activation="relu")),
                dict(name="p2",
                     type="GlobalPooling",
                     inputs=["elt"],outputs=["p2"],
                     options = dict(mode="avg")),
                dict(name="fc",
                     type="InnerProduct",
                     inputs=['p2'],
                     outputs=['fc'],
                     options = dict(
                        outputs=10,
                     )
                ),
                dict(name="prob",
                     type="SoftmaxWithLoss",
                     inputs=["fc","label"],
                     outputs=["loss"]
                )
            ]
        )
        with open("test_net.json",'w') as f:
            json.dump(self.mjs,f,indent=4)

    def forward(self, x):
        x = self.p1(F.relu(self.cnv1(x)))
        a = self.cnv2(x)
        b1 = F.relu(self.bn2a(self.cnv2a(x)))
        b = self.cnv2b(b1)
        x = F.relu(a+b)
        x = self.p2(x)
        x = self.fc(torch.flatten(x,1))
        if self.training:
            output = F.log_softmax(x, dim=1)
        else:
            output = F.softmax(x,dim=1)
        return output

    def _add_bn(self,js,fc,name,tp):
        for n,param in enumerate([fc.running_mean,fc.running_var] + list(fc.parameters())):
            name_code = '%s.%d' % (name,n)
            if tp == 'param_diff':
                v=param.grad
            elif tp == 'param' or tp == 'param_updated':
                v=param
            if v is None:
                continue
            lst = v.detach().cpu().numpy().reshape(-1).tolist()
            js.append(dict(name=name_code,type=tp,value=lst))
 
    
    def _add_fc(self,js,fc,name,tp):
        for n,param in enumerate(fc.parameters()):
            name_code = '%s.%d' % (name,n)
            if tp == 'param_diff':
                v=param.grad
            elif tp == 'param' or tp == 'param_updated':
                v=param
            lst = v.detach().cpu().numpy().reshape(-1).tolist()
            js.append(dict(name=name_code,type=tp,value=lst))
    
    def save_dp_weights(self,js,tp):
        self._add_fc(js,self.cnv1,'cnv1',tp)
        self._add_fc(js,self.cnv2,'cnv2',tp)
        self._add_fc(js,self.cnv2a,'cnv2a',tp)
        self._add_fc(js,self.cnv2b,'cnv2b',tp)
        self._add_bn(js,self.bn2a,'bn2a',tp)
        self._add_fc(js,self.fc,'fc',tp)

dataset1 = datasets.MNIST('../data', train=True, download=True,transform=True)

batch=2
ref=[]

with torch.no_grad():
    x=dataset1.data[0:batch].reshape((batch,1,28,28)).float()/255.0
    y=dataset1.targets[0:batch].reshape((batch))
    print(x.shape,x.dtype,y.shape,y.dtype)
    x_lst = x.detach().cpu().numpy().reshape(-1).tolist()
    y_lst = y.detach().cpu().numpy().reshape(-1).tolist();

    ref.append(dict(name="data",type="data",value=x_lst))
    ref.append(dict(name="label",type="data",value=y_lst))
    ref.append(dict(name="loss",type="data_diff",value=[1.0]))


model = Net2(batch)
#optimizer = optim.Adam(model.parameters(), lr = 0.001,betas=(0.9, 0.999), eps = 1e-8, weight_decay = 0.0005)
optimizer = optim.Adam(model.parameters(), lr = 0.1,betas=(0.9, 0.95), eps = 1e-3, weight_decay = 0.05)
model.train()
model.save_dp_weights(ref,"param")
optimizer.zero_grad()
out=model(x)
loss = F.nll_loss(out,y)
loss.backward()
model.save_dp_weights(ref,"param_diff")
optimizer.step()
model.save_dp_weights(ref,"param_updated")


with open('test_weights.json','w') as f:
    json.dump(ref,f,indent=2);

