import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import json


class Net2(nn.Module):
    def __init__(self,batch):
        super(Net2, self).__init__()
        self.cnv1 = nn.Conv2d(1, 32,5,padding=2)
        self.cnv2 = nn.Conv2d(32,64,5,padding=2)
        self.fc1 = nn.Linear(((28//4)**2)*64,256)
        self.fc2 = nn.Linear(256, 10)
        
        self.p1   = nn.MaxPool2d(2,stride=2)
        self.p2   = nn.MaxPool2d(2,stride=2)

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
                        channels_out = 32,
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
                        channels_out = 64,
                        kernel=list(self.cnv2.kernel_size),
                        pad=list(self.cnv2.padding),
                        activation="relu"
                    )
                ),
                dict(name="p2",
                     type="Pooling2D",
                     inputs=["cnv2"],outputs=["p2"],
                     options = dict(kernel=2,stride=2)),
                dict(name="fc1",
                     type="InnerProduct",
                     inputs=['p2'],
                     outputs=['fc1'],
                     options = dict(
                        outputs=256,
                        activation='relu'
                     )
                ),
                dict(name="fc2",
                     type="InnerProduct",
                     inputs=['fc1'],
                     outputs=['fc2'],
                     options = dict(
                        outputs=10,
                     )
                ),
                dict(name="prob",
                     type="SoftmaxWithLoss",
                     inputs=["fc2","label"],
                     outputs=["loss"]
                )
            ]
        )
        with open("test_net.json",'w') as f:
            json.dump(self.mjs,f,indent=4)

    def forward(self, x):
        x = self.p1(F.relu(self.cnv1(x)))
        x = self.p2(F.relu(self.cnv2(x)))
        x = F.relu(self.fc1(torch.flatten(x,1)))
        x = self.fc2(x)
        if self.training:
            output = F.log_softmax(x, dim=1)
        else:
            output = F.softmax(x,dim=1)
        return output

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
        self._add_fc(js,self.fc1,'fc1',tp)
        self._add_fc(js,self.fc2,'fc2',tp)

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

