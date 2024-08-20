###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
import torch
import numpy as np

def make_ref():
    bn = torch.nn.BatchNorm2d(2,affine=True)
    with torch.no_grad():
        gamma = 0.7
        beta  = 0.1
        p=list(bn.parameters())
        p[0][0]=gamma
        p[1][0]=beta
        print(p)
    pt_x=torch.randn(1,2,2,2,requires_grad=True)
    pt_y=bn(pt_x)
    pt_dy=torch.randn(1,2,2,2)
    pt_y.backward(pt_dy,retain_graph=True)
    return pt_x.detach().numpy().copy(), \
        pt_y.detach().numpy().copy(), \
        pt_dy.detach().numpy().copy(), \
        pt_x.grad.detach().numpy().copy(), \
        gamma,beta, \
        bn.running_mean.detach().numpy().copy(), \
        bn.running_var.detach().numpy().copy()


x,y,dy,dx,gamma,beta,rm,rv = make_ref()
x = x[0,0,:,:].reshape((-1))
y = y[0,0,:,:].reshape((-1))
dy=dy[0,0,:,:].reshape((-1))
dx=dx[0,0,:,:].reshape((-1))
rm=rm[0]
rv=rv[0]


print("   x",x)
print("   y",y)
print("  dy",dy)
print("  dx",dx)



eps = 1e-5
mu = np.mean(x)
sig = np.mean(x*x) - mu*mu
M = np.prod(x.shape)

my_y = (x - mu)/np.sqrt(sig + eps) * gamma + beta
print("c  y",my_y,np.sum(np.abs(y-my_y)))

#dx_hat = dy * gamma
#dsig = np.sum(dx_hat * (x - mu)) * -0.5 * np.power(sig + eps,-1.5) 
#dsig = gamma * np.sum(dy * (x - mu)) * -0.5 * np.power(sig + eps,-1.5) 

#dmu  = np.sum(dx_hat * (-1 / np.sqrt(sig + eps))) - 2* dsig * np.sum(x - mu) / M
#dmu  = np.sum(dx_hat * (-1 / np.sqrt(sig + eps)))

#my_dx = dx_hat  / np.sqrt(sig + eps)  + dsig * 2 * (x - mu) / M + dmu / M
#my_dx = dx_hat  / np.sqrt(sig + eps)  + dsig * 2 * (x - mu) / M + dmu / M

sdy = np.sum(dy)
sdyx = np.sum(dy * x)
sqrtsig = np.sqrt(sig + eps)

dsig = gamma * (sdyx  - mu * sdy) * -0.5 / (sqrtsig * sqrtsig * sqrtsig)
dmu  = - sdy * gamma / sqrtsig
#my_dx = dy * (gamma / sqrtsig)  + dsig * 2 * (x - mu) / M + dmu / M
my_dx = dy * (gamma / sqrtsig)  + (dsig * 2 / M ) * x - (dsig * 2 * mu) / M + dmu / M


print("c dx",my_dx,np.sum(np.abs(dx-my_dx)))
my_rm = 0.1 * np.mean(x)
moved_x = x - np.mean(x)
#my_var = np.mean(moved_x*moved_x)
my_var = (np.mean(x*x) - np.mean(x)**2)
my_var = M / ( M - 1) * my_var

my_rv = 0.9 +my_var * 0.1
print("c rm",my_rm,rm,rm-my_rm)
print("c rv",my_rv,rv,rv-my_rv)

