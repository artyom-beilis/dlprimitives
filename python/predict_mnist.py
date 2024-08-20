###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
import sys
import numpy as np
import dlprim as dp
import PIL
from PIL import Image

ctx=dp.Context(sys.argv[1])
print("Using",ctx.name)
q=ctx.make_execution_context()

net = dp.Net(ctx)
net.load_from_json_file('../build/mnist_dp.json')
net.setup()
net.load_parameters('../build/snap_5.dlp',False)
data = net.tensor('data')

imgs = sys.argv[2:]
batch = data.shape[0]
print("Batch size:",batch)
inp = np.zeros((batch,1,28,28),dtype=np.float32)
scores = np.zeros((batch,10),dtype=np.float32)

def load_batch(index,batch,inp):
    images = sys.argv[2+index:2+index+batch]
    for i,img in enumerate(images):
        pic = Image.open(img)
        np.copyto(inp[i],(np.array(pic).astype(np.float32)*(1/255.0)).reshape(1,28,28))
    return len(images)

for index in range(0,len(sys.argv[2:]),batch):
    n=load_batch(index,batch,inp)
    data.to_device(inp,q)
    net.forward(q,False)
    net.tensor('prob').to_host(scores,q)
    for k in range(n):
        print(sys.argv[2+index+k],np.argmax(scores[k]))


