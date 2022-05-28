###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
import numpy as np
import dlprim as dp

ctx=dp.Context("0:0")
print(ctx.name)
q=ctx.make_execution_context()
#t1 = dp.Tensor(ctx,dp.Shape(5,10))
t1 = dp.Tensor(ctx,dp.Shape(5,10),dp.float32)
print(t1.shape)
print(t1.dtype)
a = np.random.randn(5,10).astype(np.float32)
print(a.shape)
b = np.zeros((5,10),dtype=np.float32)

t1.to_device(a,q)
t1.to_host(b,q)

print(a)
print(b)

net = dp.Net(ctx)
net.load_from_json_file('../build/mnist_dp.json')
net.setup()
net.load_parameters('../build/snap_5.dlp',False)
print(net.input_names)
print(net.output_names)
data = net.tensor('data')
data.reshape(dp.Shape(2,1,28,28));
net.reshape()
data.to_device(np.random.randn(2,1,28,28).astype(np.float32),q)
net.forward(q,False)
res = net.tensor(net.output_names[0])
print(res.shape)
np_res = np.zeros((2,10),dtype=np.float32)
res.to_host(np_res,q)
print(np_res)

