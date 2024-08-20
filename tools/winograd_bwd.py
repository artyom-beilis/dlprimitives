###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
import numpy as np
import torch
A = np.array([[1,1,1,0],[0,1,-1,-1]]).T.astype(np.float32).copy()
B = np.array([[1,0,-1,0],
              [0,1,1,0],
              [0,-1,1,0],
              [0,1,0,-1]]).T.astype(np.float32).copy()
G = np.array([[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]]).astype(np.float32)


cnv = torch.nn.Conv2d(1,1,3,padding=0,bias=False)
pt_x = torch.randn(1,1,4,4,requires_grad=True)
x=pt_x.detach().numpy().reshape((4,4)).copy()

pt_k = list(cnv.parameters())[0]
k=pt_k.detach().numpy().reshape((3,3)).copy()

#####################
x16 = np.dot(np.dot(B.T,x),B)
k16 = np.dot(np.dot(G,k),G.T)
y16 = k16*x16
y4  = np.dot(np.dot(A.T,y16),A)

pt_y = cnv(pt_x)
pt_y_ref = pt_y.detach().numpy().reshape((2,2)).copy()
pt_dy = torch.randn(1,1,2,2)
dy = pt_dy.detach().numpy().reshape((2,2)).copy()
pt_y.backward(pt_dy,retain_graph=True)

pt_dx = pt_x.grad
dx_ref = pt_dx.detach().reshape((4,4)).numpy().copy()

#======================
print(A)
dy16 = np.dot(np.dot(A,dy),A.T)
#print(np.dot(np.dot(A,[[ 3, 7],[-5, 3]]),A.T))
dx16 = k16 * dy16
dx = np.dot(np.dot(B,dx16),B.T)

#========================
dk16 = dy16 * x16
dk   = np.dot(np.dot(G.T,dk16),G)
dk_ref = pt_k.grad.detach().numpy().reshape((3,3)).copy()

print("Y")
print(y4);
print(pt_y_ref)
print(np.sum(np.abs(y4 - pt_y_ref)))
print("DX")
print(dx)
print(dx_ref)
print(np.sum(np.abs(dx-dx_ref)))
print("DK")
print(dk)
print(dk_ref)
print(np.sum(np.abs(dk - dk_ref)))
