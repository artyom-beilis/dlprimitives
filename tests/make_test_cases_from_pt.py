###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
import torch
import json
import numpy as np
import argparse
import random

def save_report(r):
    name,js = r
    with open('test_case_%s.json' %name,'w') as f:
        json.dump(js,f,indent=2)

def make_softmax_loss():
    report = {
        "operator" : "SoftmaxWithLoss",
        "tests" : [
            {
                "train" : True,
                "options" : {},
                "setup_tensors" : [ {"shape":[10,50]}, {"shape":[10],"requires_grad" : False} ],
                "output_tensors" : [ {"shape":[1]} ],
                "workspace": 0,
                "cases" : []
            }
        ]
    }
    cases = report["tests"][0]["cases"]
    sm = torch.nn.LogSoftmax(dim=1)
    for b in [1,2,5,10,50]:
        for f in [2,5,31,32,33,63,64,65]:
            case = {
                "in_shapes"  : [[b,f],[b]],
                "out_shapes" : [[1]],
            }
            inp = torch.randn(b,f,requires_grad=True)
            lbl = torch.randint(f,size=(b,))
            out = torch.nn.NLLLoss()(sm(inp),lbl)
            dl  = torch.rand(1).item() * 0.5 + 0.5
            out.backward(torch.tensor(dl),retain_graph=True);
            case["in_tensors"] = [inp.reshape((-1,)).tolist(),lbl.reshape((-1,)).tolist()]
            case["out_tensors"] = [out.reshape((-1,)).tolist()]
            case["out_diffs"] = [[dl]]
            case["in_diffs"] = [inp.grad.reshape((-1,)).tolist(),False]
            cases.append(case)
    return report

def make_nll_loss():
    report = {
        "operator" : "NLLLoss",
        "tests" : []
    }
    tests = report["tests"]
    for reduction in ['none','mean','sum']:
        cases = []
        test =  {
            "train" : True,
            "options" : {'reduce':reduction},
            "setup_tensors" : [ {"shape":[10,50]}, {"shape":[10],"requires_grad" : False} ],
            "output_tensors" : [ {"shape": ([1] if reduction != 'none' else [10])} ],
            "workspace": 0,
            "cases" : cases
        }
        tests.append(test)
        print("-",reduction)
        loss = torch.nn.NLLLoss(reduction=reduction)
        for b in [1,2,5,10,50]:
            for f in [2,5,31,32,33,63,64,65]:
                case = {
                    "in_shapes"  : [[b,f],[b]],
                    "out_shapes" : [[1] if reduction != 'none' else [b]],
                }
                inp = torch.randn(b,f,requires_grad=True)
                lbl = torch.randint(f,size=(b,))
                out = loss(inp,lbl)
                with torch.no_grad():
                    dl  = torch.rand(*case['out_shapes'][0]) * 0.5 + 0.5
                    if reduction != 'none':
                        val = dl.item();
                        dl_print= [dl.item()]
                        dl=torch.tensor(val)
                    else:
                        dl_print = dl.reshape((-1,)).tolist()
                out.backward(dl,retain_graph=True);
                case["in_tensors"] = [inp.reshape((-1,)).tolist(),lbl.reshape((-1,)).tolist()]
                case["out_tensors"] = [out.reshape((-1,)).tolist()]
                case["out_diffs"] = [dl.reshape((-1,)).tolist()]
                case["in_diffs"] = [inp.grad.reshape((-1,)).tolist(),False]
                cases.append(case)
    return report

def make_mse_loss():
    report = {
        "operator" : "MSELoss",
        "tests" : []
    }
    tests = report["tests"]
    for reduction in ['none','mean','sum']:
        cases = []
        test =  {
            "train" : True,
            "options" : {'reduce':reduction},
            "setup_tensors" : [ {"shape":[10]}, {"shape":[10]} ],
            "output_tensors" : [ {"shape": ([1] if reduction != 'none' else [10])} ],
            "workspace": 0,
            "cases" : cases
        }
        tests.append(test)
        print("-",reduction)
        loss = torch.nn.MSELoss(reduction=reduction)
        for s in [[2],[2,3],[2,3,4]]:
            case = {
                "in_shapes"  : [s,s],
                "out_shapes" : [[1] if reduction != 'none' else s],
            }
            inp1 = torch.randn(*s,requires_grad=True)
            inp2 = torch.randn(*s,requires_grad=True)
            out = loss(inp1,inp2)
            with torch.no_grad():
                dl  = torch.rand(*case['out_shapes'][0]) * 0.5 + 0.5
                if reduction != 'none':
                    val = dl.item();
                    dl_print= [dl.item()]
                    dl=torch.tensor(val)
                else:
                    dl_print = dl.reshape((-1,)).tolist()
            out.backward(dl,retain_graph=True);
            case["in_tensors"] = [inp1.reshape((-1,)).tolist(),inp2.reshape((-1,)).tolist()]
            case["out_tensors"] = [out.reshape((-1,)).tolist()]
            case["out_diffs"] = [dl.reshape((-1,)).tolist()]
            case["in_diffs"] = [inp1.grad.reshape((-1,)).tolist(),inp2.grad.reshape((-1,)).tolist()]
            cases.append(case)
    return report


def make_log_softmax():
    return make_softmax(True)

def make_softmax(log=False):
    train = True #log
    report = {
        "operator" : "Softmax",
        "tests" : [
            {
                "train" : train,
                "options" : {"log":log},
                "setup_tensors" : [ {"shape":[10,50]} ],
                "output_tensors" : [ {"shape":[10,50]} ],
                "workspace": 0,
                "cases" : []
            }
        ]
    }
    cases = report["tests"][0]["cases"]
    if log:
        sm = torch.nn.LogSoftmax(dim=1)
    else:
        sm = torch.nn.Softmax(dim=1)
    for d2 in [[],[1],[5],[10]]:
        for b in [1,2,5,10,50,100,120]:
            for f in [1,5,31,32,33,63,64,65,100,1000,2000]:
                case = {
                    "in_shapes"  : [[b,f] + d2],
                    "out_shapes" : [[b,f] + d2],
                }
                if b <= 5 and f <= 32:
                    inp = torch.randn(b,f,*d2)
                    inp.requires_grad = train
                    out = sm(inp)
                    case["in_tensors"] = [inp.reshape((-1,)).tolist()]
                    case["out_tensors"] = [out.reshape((-1,)).tolist()]
                    if train:
                        dout = torch.randn(out.shape)
                        out.backward(dout,retain_graph=True)
                        case["out_diffs"] = [dout.reshape((-1,)).tolist()]
                        case["in_diffs"] = [inp.grad.reshape((-1)).tolist()]
                else:
                    case["use_cpu_reference"]=True
                if f > 1000:
                    case['eps'] = 3e-3
                elif f > 100:
                    case['eps'] = 1e-3
                cases.append(case)
    return report

def make_eltwise():
    report = {
        "operator" : "Elementwise",
        "tests" : []
    }
    tests = report["tests"]
    for op,mop in [ ("sum", lambda x,y:x+y),
                    ("prod",lambda x,y:x*y),
                    ("max" ,lambda x,y:torch.maximum(x,y)) ]:
        for c1 in [ -1.5, 2.0 ]:
            for c2 in [ 1.2 ]:
                for act,mact in [ ("identity",lambda x:x),
                             ("relu",torch.nn.ReLU()) ]:
                    cases=[]
                    test = {
                        "train":True,
                        "options" : {
                            "operation":op,
                            "activation": act,
                            "coef1":c1,
                            "coef2":c2,
                        },
                        "setup_tensors" : [ {"shape":[10,50]},{"shape":[10,50]} ],
                        "output_tensors" : [ {"shape":[10,50]} ],
                        "workspce": 0,
                        "cases": cases
                    }
                    tests.append(test)
                    final_op = lambda x,y: mact(mop(c1*x,c2*y))
                    for s1,s2 in [([5,2],[5,2]),
                                  ([5,2],[5,1]),
                                  ([5,2],[2]),
                                  ([2,3,4],[2,3,4]),
                                  ([2,3,4],[4]),
                                  ([2,3,4],[3,1]),
                                  ([2,3,4],[2,1,1]),
                                  ([1,3,1],[2,3,4]),

                                  ([2,3,4,5],[2,3,4,5]),
                                  ([2,3,4,5],[4,1]),
                                  ([2,3,4,5],[3,1,1]),
                                  ([2,3,4,5],[2,1,1,1]),
                                  ([3,1,5],[2,3,4,5]),

                                  ([2,3,4,5,6],[2,3,4,5,6]),
                                  ([2,3,4,5,6],[4,1,1]),
                                  ([2,3,4,5,6],[3,1,1,1]),
                                  ([2,3,4,5,6],[2,1,1,1,1]),
                                  ([3,1,1,1],[2,3,4,5,6]),

                                  ([32,6,16,16],[32,6,16,16]),
                                  ([32,1,16,16],[6,16,1]),
                                  ([32,6,16,16],[6,1,1])]:
                                  #([128,256,16,16]),[10023]]:
                        if np.prod(s1) < 1000 and np.prod(s2) < 1000:
                            print("- generating test for ",s1,s2);
                            a = torch.randn(*s1,requires_grad=True)
                            b = torch.randn(*s2,requires_grad=True)
                            c = final_op(a,b)
                            case = dict(in_shapes = [ s1, s2 ] ,out_shapes = [list(c.shape)])
                            dc = torch.randn(*c.shape)
                            c.backward(dc,retain_graph=True)
                            case["in_tensors"] = [a.reshape((-1,)).tolist(),b.reshape((-1,)).tolist()]
                            case["out_tensors"] = [c.reshape((-1,)).tolist()]
                            case["out_diffs"] = [dc.reshape((-1,)).tolist()]
                            case["in_diffs"] = [a.grad.reshape((-1)).tolist(),b.grad.reshape((-1)).tolist()]
                        else:
                            print("- cpu test for ",s1,s2);
                            case["use_cpu_reference"]=True
                        cases.append(case)
    return report

def make_slice():
    report = {
        "operator" : "Slice",
        "tests" : []
    }
    tests = report["tests"]
    tc = [
        (0, 1,3,[ [10,5],[15,2],[20,5] ]),
        (1, 2,7,[ [2,10,5],[5,10,1] ]),
        (1, 2,7,[ [2,10],[5,10] ])
    ]
    for dim,begin,end,shapes in tc:
        cases=[]
        out_shape = shapes[0][:]
        out_shape[dim]  = end - begin
        test = {
            "train" : True,
            "options" : {
                "dim": dim,
                "begin":begin,
                "end":end,
            },
            "setup_tensors" : [ {"shape": shapes[0]} ],
            "output_tensors" : [ {"shape": out_shape} ],
            "workspce": 0,
            "cases": cases
        }
        tests.append(test)
        for shape in shapes:
            inp_shape = shape
            out_shape = inp_shape[:]
            out_shape[dim]  = end - begin
            case = dict(in_shapes = [inp_shape] ,out_shapes = [out_shape])
            
            a = torch.randn(*shape,requires_grad=True)
            if len(shape) == 2 and dim == 0:
                c = a[begin:end,:]
            elif len(shape)==2 and dim == 1:
                c = a[:,begin:end]
            elif len(shape)==3 and dim == 1:
                c = a[:,begin:end,:]
            else:
                assert not "get there"
            dc = torch.randn(c.shape)
            c.backward(dc,retain_graph=True)
            assert tuple(c.shape) == tuple(out_shape),"%s %s" % (str(c.shape),str(out_shape))
            case["in_tensors"]  = [a.reshape((-1,)).tolist()]
            case["out_tensors"] = [c.reshape((-1,)).tolist()]
            case["out_diffs"] = [dc.reshape((-1,)).tolist()]
            case["in_diffs"] = [a.grad.reshape((-1)).tolist()]
            cases.append(case)
    return report


def make_concat():
    report = {
        "operator" : "Concat",
        "tests" : []
    }
    tests = report["tests"]
    tc = [
        (0, [ [[10,5],[10,5]],
              [[30,5],[30,5]],
              [[1,1], [1,1 ]] ] ),
        (0, [ [[10,5],[5,5],    [15,5]],
              [[3,20],[7,20],   [10,20]] ]),
        (0, [ [[10,5],[5,5],[3,5],    [18,5]],
              [[3,20],[7,20],[1,20],   [11,20]] ]),
        (1, [ [[10,5],[10,5]],
              [[30,5],[30,5]],
              [[1,1], [1,1 ]] ] ),
        (1, [ [[10,5],[10,5],    [10,10]],
              [[15,2],[15,1],    [15,3 ]] ]),
        (1, [ [[10,5,2],[10,5,2],[10,10,2]],
              [[15,2,4],[15,1,4],    [15,3,4 ]] ]),

    ]
    for dim,tmpl in tc:
        cases=[]
        test = {
            "train" : True,
            "options" : {
                "dim": dim,
            },
            "setup_tensors" : [ {"shape":v} for v in tmpl[0][0:-1]  ],
            "output_tensors" : [ {"shape":tmpl[0][-1]} ],
            "workspce": 0,
            "cases": cases
        }
        tests.append(test)
        for shapes in tmpl:
            inp_shapes = shapes[0:-1]
            out_shape = shapes[-1]
            case = dict(in_shapes = inp_shapes ,out_shapes = [out_shape])
            
            ins = []
            for s in inp_shapes:
                a = torch.randn(*s,requires_grad=True)
                ins.append(a)
            c = torch.cat(ins,dim=dim)
            dc = torch.randn(c.shape)
            c.backward(dc,retain_graph=True)
            assert tuple(c.shape) == tuple(out_shape)
            case["in_tensors"]  = [a.reshape((-1,)).tolist() for a in ins]
            case["out_tensors"] = [c.reshape((-1,)).tolist()]
            case["out_diffs"] = [dc.reshape((-1,)).tolist()]
            case["in_diffs"] = [a.grad.reshape((-1)).tolist() for a in ins]
            cases.append(case)
    return report


def make_pointwise(op,opset):
    report = {
        "operator" : op,
        "tests" : []
    }
    tests = report["tests"]
    for cfg,op in opset:
        cases=[]
        test = {
            "train" : True,
            "options" : cfg,
            "setup_tensors" : [ {"shape":[10,50]}  ],
            "output_tensors" : [ {"shape":[10,50]} ],
            "workspce": 0,
            "cases": cases
        }
        tests.append(test)
        for shape in [[5,2],[32,6,16,16],[128,256,16,16],[10023]]:
            case = dict(in_shapes = [ shape ] ,out_shapes = [shape])
            if np.prod(shape) < 100:
                a = torch.randn(*shape)
                a.requires_grad = True
                c = op(a)
                dc = torch.randn(c.shape)
                c.backward(dc,retain_graph=True)
                case["in_tensors"]  = [a.reshape((-1,)).tolist()]
                case["out_tensors"] = [c.reshape((-1,)).tolist()]
                case["out_diffs"] = [dc.reshape((-1,)).tolist()]
                case["in_diffs"] = [a.grad.reshape((-1)).tolist()]
            else:
                case["use_cpu_reference"]=True
            cases.append(case)
    return report

def make_hardtanh():
    return make_pointwise("Hardtanh", [
        ( dict(min_val=-0.1,max_val=1.1), torch.nn.Hardtanh(min_val=-0.1,max_val=1.1) ),
        ( dict(min_val=0,max_val=6), torch.nn.ReLU6() )
    ])

def make_abs():
    return make_pointwise("Abs", [
        ( dict(), torch.abs )
    ])

def make_threshold():
    return make_pointwise("Threshold", [
        (dict(), lambda x: (torch.nn.Hardtanh(0,1e-20)(x))*1e20),
        (dict(threshold=0.5), lambda x: (torch.nn.Hardtanh(0,1e-20)(x-0.5))*1e20)
    ])


def make_activation():
    report = {
        "operator" : "Activation",
        "tests" : []
    }
    tests = report["tests"]
    for act,op in [ ("relu", torch.nn.ReLU()),
                    ("tanh", torch.nn.Tanh()),
                    ("sigmoid",torch.nn.Sigmoid()),
                    ("identity", (lambda x:x) ),
                    ("relu6", torch.nn.ReLU6() ) ]:
        cases=[]
        test = {
            "train" : True,
            "options" : {
                "activation": act,
            },
            "setup_tensors" : [ {"shape":[10,50]}  ],
            "output_tensors" : [ {"shape":[10,50]} ],
            "workspce": 0,
            "cases": cases
        }
        tests.append(test)
        for shape in [[5,2],[32,6,16,16],[128,256,16,16],[10023]]:
            case = dict(in_shapes = [ shape ] ,out_shapes = [shape])
            if np.prod(shape) < 100:
                a = torch.randn(*shape)*16
                a.requires_grad = True
                c = op(a)
                dc = torch.randn(c.shape)
                c.backward(dc,retain_graph=True)
                case["in_tensors"]  = [a.reshape((-1,)).tolist()]
                case["out_tensors"] = [c.reshape((-1,)).tolist()]
                case["out_diffs"] = [dc.reshape((-1,)).tolist()]
                case["in_diffs"] = [a.grad.reshape((-1)).tolist()]
            else:
                case["use_cpu_reference"]=True
            cases.append(case)
    return report

def make_interpolation():
    report = {
        "operator" : "Interpolation",
        "tests" : []
    }
    tests = report["tests"]
    for method,align in [('bilinear',False),('bilinear',True),('nearest',None),('nearest-exact',None)]:
        for xs in [(4,8),(7,9),(16,16)]:
            cfgs = []
            for scale in [ (2,1),(2,2),(3.4,3.3),(0.5,0.7)]:
                cfgs.append((None,scale))
            for ys in [(8,8),(15,16),(4,4)]:
                cfgs.append((ys,None))
            for cfg in cfgs:
                cases=[]
                tin = torch.randn(1,1,*xs)
                op = lambda x:torch.nn.functional.interpolate(x,size=cfg[0],scale_factor = cfg[1],mode=method,align_corners = align)
                tout = op(tin)
                test = {
                    "train" : True,
                    "options" : {
                        "method" : method,
                        "out_h": -1 if cfg[0] is None else cfg[0][0],
                        "out_w": -1 if cfg[0] is None else cfg[0][1],
                        "scale_y": -1 if cfg[1] is None else cfg[1][0],
                        "scale_x": -1 if cfg[1] is None else cfg[1][1],
                        "align_corners" : False if align is None else align,
                    },
                    "setup_tensors" : [ { "shape" : list(tin.shape) } ],
                    "output_tensors": [ { "shape" : list(tout.shape) } ],
                    "workspce": 0,
                    "cases": cases
                }
                print(test["options"])
                tests.append(test)
                for s in [[1,1,*xs],[2,3,*xs],[4,4,*xs],[64,128,*xs]]:
                    print("- ",s)
                    tin = torch.randn(s,requires_grad=True)
                    tout = op(tin)
                    dout = torch.randn(tout.shape)
                    tout.backward(dout,retain_graph=True)
                    case = dict(in_shapes = [ list(tin.shape)] ,out_shapes = [list(tout.shape)])
                    if np.prod(s) < 200:
                        case["in_tensors"] = [tin.reshape((-1,)).tolist()]
                        case["out_tensors"] = [tout.reshape((-1,)).tolist()]
                        case["out_diffs"] = [dout.reshape((-1,)).tolist()]
                        case["in_diffs"] = [tin.grad.reshape((-1)).tolist()]
                    else:
                        case["use_cpu_reference"]=True
                        case["eps"] = 1e-4
                    cases.append(case)
    return report



def make_pooling2d():
    report = {
        "operator" : "Pooling2D",
        "tests" : []
    }
    tests = report["tests"]
    for krn,pad,stride in \
        [ (2,0,2),
          (3,1,1),
          ([2,4],[1,0],[1,2]),
          ([3,5],[1,2],[3,4]) 
        ]:
          for tp,inc_pad,op,cm in [ 
                ("max",False,torch.nn.MaxPool2d(krn,stride=stride,padding=pad),False),
                ("avg",False,torch.nn.AvgPool2d(krn,stride=stride,padding=pad,count_include_pad=False),False),
                ("avg",True, torch.nn.AvgPool2d(krn,stride=stride,padding=pad,count_include_pad=True),False), 
                ("max",False,torch.nn.MaxPool2d(krn,stride=stride,padding=pad,ceil_mode=True),True),
                ("avg",False,torch.nn.AvgPool2d(krn,stride=stride,padding=pad,count_include_pad=False,ceil_mode=True),True),
                ("avg",True, torch.nn.AvgPool2d(krn,stride=stride,padding=pad,count_include_pad=True,ceil_mode=True),True) 
            ]:
            cases=[]
            tin = torch.randn(4,16,32,32)
            tout = op(tin)
            test = {
                "train" : True,
                "options" : {
                    "mode": tp,
                    "kernel":krn,
                    "pad" : pad,
                    "stride" : stride,
                    "ceil_mode" : cm,
                    "count_include_pad": inc_pad
                },
                "setup_tensors" : [ { "shape" : list(tin.shape) } ],
                "output_tensors": [ { "shape" : list(tout.shape) } ],
                "workspce": 0,
                "cases": cases
            }
            print(test["options"])
            tests.append(test)
            for s in [[2,3,5,5],[1,1,6,6],[1,1,7,7],[1,1,8,8],
                      [8,32,64,65],[8,256,247,247]]:
                print("- ",s)
                tin = torch.randn(s,requires_grad=True)
                tout = op(tin)
                dout = torch.randn(tout.shape)
                tout.backward(dout,retain_graph=True)
                case = dict(in_shapes = [ list(tin.shape)] ,out_shapes = [list(tout.shape)])
                if np.prod(s) < 200:
                    case["in_tensors"] = [tin.reshape((-1,)).tolist()]
                    case["out_tensors"] = [tout.reshape((-1,)).tolist()]
                    case["out_diffs"] = [dout.reshape((-1,)).tolist()]
                    case["in_diffs"] = [tin.grad.reshape((-1)).tolist()]
                else:
                    case["use_cpu_reference"]=True
                cases.append(case)
    return report

def make_reduction():
    report = {
        "operator" : "Reduction",
        "tests" : []
    }
    tests = report["tests"]
    for keepdims in [False,True]:
        for dims,dim_list,cfg in [
                (1,[0],{}),
                (1,[0],{"dims":[0]}),
                (1,[0],{"start_axis":0}),
                (2,[1],{"start_axis":1}),
                (2,[1],{"dims":[1]}),
                (2,[0,1],{"start_axis":0}),
                (2,[0,1],{"dims":[0,1]}),
                (3,[2],{"start_axis":-1}),
                (3,[1,2],{"start_axis":1}),
                (3,[1,2],{"start_axis":1,"output_scale":0.2}),
                (3,[1,2],{"dims":[-1,-2]}),
                (3,[0,2],{"dims":[0,-1]}),
                (4,[1,2,3],{"start_axis":1}),
                (4,[1,2],{"dims":[1,2]}),
                (5,[2,3,4],{"start_axis":2}),
                (5,[1,3,4],{"dims":[1,3,4]}),
            ]:

            scale = cfg.get("output_scale",1)
            torch_keep_dims = keepdims
            for method,op in [
                    ("sum",lambda x: scale * torch.sum(x,dim_list,keepdims = torch_keep_dims) ),
                    ("sumsq", lambda x: scale * torch.sum(x*x,dim_list,keepdims = torch_keep_dims) ),
                    ("abssum", lambda x: scale * torch.sum(torch.abs(x),dim_list,keepdims = torch_keep_dims) ),
                    ("mean", lambda x: scale * torch.mean(x,dim_list,keepdims = torch_keep_dims) )
                ]:
                cases=[]
                tin=torch.randn({1:(5),2:(2,3),3:(2,3,4),4:(2,3,4,5),5:(2,3,4,5,6)}[dims])
                tout = op(tin)
                os = list(tout.shape)
                if not os:
                    os = [1]
                test = {
                    "train" : True,
                    "options" : {},
                    "setup_tensors" : [ { "shape" : list(tin.shape) } ],
                    "output_tensors": [ { "shape" : os } ],
                    "workspce": 0,
                    "cases": cases
                }
                test["options"].update(cfg)
                test["options"]["method"] = method
                test["options"].update(dict(keep_dim = keepdims))
                print(test["options"])
                tests.append(test)
                for s in [  (5,),(100,),
                            (2,3),(200,300),
                            (2,3,4),(20,30,40),
                            (2,3,4,5),(20,30,40,50),
                            (2,3,4,5,6),(20,30,40,50,60) ]:
                    if len(s) != dims:
                        continue
                    print("- ",s)
                    tin = torch.randn(s,requires_grad=True)
                    tout = op(tin)
                    dy = torch.randn(tout.shape);
                    tout.backward(dy,retain_graph=True);
                    out_shape = list(tout.shape)
                    if not out_shape:
                        out_shape = [1]
                    case = dict(in_shapes = [ list(tin.shape)] ,out_shapes = [out_shape])
                    if np.max(s) < 10:
                        case["in_tensors"] = [tin.reshape((-1,)).tolist()]
                        case["out_tensors"] = [tout.reshape((-1,)).tolist()]
                        case["out_diffs"] = [dy.reshape((-1,)).tolist()]
                        case["in_diffs"] = [tin.grad.reshape((-1)).tolist()]
                    else:
                        case["use_cpu_reference"]=True
                        if len(s) <= 4:
                            case["eps"] = 0.01
                        else:
                            case["eps"] = 0.1
                    cases.append(case)
    return report



def make_global_pooling():
    report = {
        "operator" : "GlobalPooling",
        "tests" : []
    }
    tests = report["tests"]
    for tp,op in [ 
            ("max",torch.nn.AdaptiveMaxPool2d((1,1))),
            ("avg",torch.nn.AdaptiveAvgPool2d((1,1))),
        ]:
        cases=[]
        tin = torch.randn(4,16,32,32)
        tout = op(tin)
        test = {
            "train" : True,
            "options" : {
                "mode": tp,
            },
            "setup_tensors" : [ { "shape" : list(tin.shape) } ],
            "output_tensors": [ { "shape" : list(tout.shape) } ],
            "workspce": 0,
            "cases": cases
        }
        print(test["options"])
        tests.append(test)
        for s in [[2,3,5,5],[1,1,6,6],[1,1,7,7],[1,1,8,8],
                  [8,32,64,65],[8,256,247,247],[5,1,1000,1000]]:
            print("- ",s)
            tin = torch.randn(s,requires_grad=True)
            tout = op(tin)
            dy = torch.randn(s[0],s[1],1,1);
            tout.backward(dy,retain_graph=True);
            case = dict(in_shapes = [ list(tin.shape)] ,out_shapes = [list(tout.shape)])
            if np.prod(s) < 200:
                case["in_tensors"] = [tin.reshape((-1,)).tolist()]
                case["out_tensors"] = [tout.reshape((-1,)).tolist()]
                case["out_diffs"] = [dy.reshape((-1,)).tolist()]
                case["in_diffs"] = [tin.grad.reshape((-1)).tolist()]
            else:
                case["use_cpu_reference"]=True
            cases.append(case)
    return report

def make_batchnorm():
    report = {
        "operator" : "BatchNorm",
        "tests" : []
    }
    tests = report["tests"]
    for features,dims,affine,extra,cpu_as_ref in \
                  [ (5,4,True,{},False),
                    (5,4,False,{},False),
                    (7,4,True,{},False),
                    (6,3,True,{"eps":1e-2},False),
                    (7,2,True,{"eps":1e-2},False),
                    (6,4,False,{},False),
                    (30,4,True,dict(momentum=0.7,eps=0.01),False),
                    (300,4,True,{},True),
                    ]:
        bn = {4:torch.nn.BatchNorm2d,3:torch.nn.BatchNorm1d,2:torch.nn.BatchNorm1d}
        op = (bn[dims])(features,eps=extra.get("eps",1e-5),momentum=extra.get('momentum',0.1),affine=affine)
        params = [op.running_mean,op.running_var] + list(op.parameters())
        if affine:
            with torch.no_grad():
                params[2][:]=torch.randn(features)[:]
                params[3][:]=torch.randn(features)[:]
        cases=[]
        test = {
            "init" : "small_frac",
            "train" : True,
            "options" : {
                "features": features,
                "affine" : affine
            },
            "setup_tensors" : [ { "shape" : [64,features,256,256] } ],
            "output_tensors": [ { "shape" : [64,features,256,256] } ],
            "param_specs":  [ { "shape" : list(p.shape), "trainable" : i >= 2  } for i,p in enumerate(params) ],
            "cases": cases
        }
        test['param_tensors'] = [ p.reshape((-1,)).tolist() for p in params ]
        test['options'].update(extra)
        pred_param=True
        tests.append(test)
        set_a = [  (True, [2,features,1,1]),
                   (True, [2,features,1,1]),
                   (True, [2,features,1,1]),
                   (False,[2,features,1,1]),
                   (False,[2,features,1,1]),
                   (True, [2,features,4,4]),
                   (True, [2,features,4,4]),
                   (True, [2,features,4,4]),
                   (False,[2,features,4,4]),
                   (False,[2,features,4,4]),
                   (True,[5,features,5,7]),
                   (True,[5,features,5,7]),
                   (False,[5,features,5,7]),
                   (False,[5,features,5,7]) ] 
        set_b = [ (True,(17,features,35,24)),
                  (True,(17,features,35,24)),
                  (False,(17,features,35,24)),
                  (True,(32,features,64,64)),
                  (True,(32,features,64,64)),
                  (False,(32,features,64,64)),
                  (True,(32,features,64,64))]
        for train,sin in (set_b if cpu_as_ref else set_a):
            s=sin[0:dims]
            if train:
                op.train()
            else:
                op.eval()
            if not cpu_as_ref:
                tin = torch.randn(s)
                tin.requires_grad = True
                tout = op(tin)
                dtout = torch.randn(tout.shape)
                tout.backward(dtout,retain_graph=True)
            case = dict(in_shapes = [ list(s)] ,out_shapes = [list(s)],test_mode = not train,eps=1e-4)
            if not cpu_as_ref:
                case["in_tensors"] = [tin.reshape((-1,)).tolist()]
                case["out_tensors"] = [tout.reshape((-1,)).tolist()]
                case["out_diffs"] = [dtout.reshape((-1,)).tolist()]
                case["in_diffs"] = [tin.grad.reshape((-1)).tolist()]
                case["params_diffs"] = [ False if p.grad is None else p.grad.reshape((-1,)).tolist() for p in params ]
                case["double_check"] = False
                case["parameters_check"] = [ False if p.grad is not None else p.reshape((-1,)).tolist() for p in params ]
                print("- ",s,features,"predefined")
            else:
                case["use_cpu_reference"]=True
                print("- ",s,features,"cpu as ref")
            cases.append(case)
            for p in params:
                if p.grad is not None:
                    p.grad*=0
    return report

def make_param():
    report = {
        "operator" : "Parameter",
        "tests" : []
    }
    tests = report["tests"]
    par = torch.randn((2,3,4))
    par.requires_grad = True
    params = [par]
    cases=[]
    tout = par + 0
    test = {
        "train" : True,
        "options" : {
            "shape": [2,3,4],
            "dtype" : "float"
        },
        "setup_tensors" : [ ],
        "output_tensors": [ { "shape" : list(tout.shape) } ],
        "param_specs":  [ { "shape" : list(p.shape) } for p in params ],
        "workspce": 0,
        "cases": cases
    }
    test['param_tensors'] = [ p.reshape((-1,)).tolist() for p in params ]
    print(test["options"])
    tests.append(test)
    dtout = torch.randn(tout.shape)
    tout.backward(dtout,retain_graph=True)
    case = dict(in_shapes = [] ,out_shapes = [list(tout.shape)])
    case["in_tensors"] = []
    case["out_tensors"] = [tout.reshape((-1,)).tolist()]
    case["out_diffs"] = [dtout.reshape((-1,)).tolist()]
    case["in_diffs"] = []
    case["params_diffs"] = [ p.grad.reshape((-1,)).tolist() for p in params ]
    cases.append(case)
    return report
    
def make_inner_product():
    report = {
        "operator" : "InnerProduct",
        "tests" : []
    }
    tests = report["tests"]
    for batch in [1,128,512]:
        for inp,out in \
            [ (5,11),
              (1024,1024),
              (5,1024),
              (1024,5),
              (4096,4096),
              (500,1000) ]:
              for bias in [True,False]:
                op = torch.nn.Linear(inp,out,bias=bias) 
                for act,mact in [ ("identity",lambda x:x), ("relu",torch.nn.ReLU()) ]:
                    params = list(op.parameters())
                    cases=[]
                    tin = torch.randn(batch,inp)
                    tout = op(tin)
                    test = {
                        "train" : True,
                        "init" : "small_frac",
                        "options" : {
                            "inputs": inp,
                            "outputs" : out,
                            "activation": act,
                            "bias" : bias
                        },
                        "setup_tensors" : [ { "shape" : list(tin.shape) } ],
                        "output_tensors": [ { "shape" : list(tout.shape) } ],
                        "param_specs":  [ { "shape" : list(p.shape) } for p in params ],
                        "workspce": 0,
                        "cases": cases
                    }
                    if inp * out < 100:
                        test['param_tensors'] = [ p.reshape((-1,)).tolist() for p in params ]
                    else:
                        test['random_params'] = True

                    print(test["options"])
                    tests.append(test)
                    final_op = lambda x: mact(op(x))
                    for s in [[1,inp],[2,inp],[8,inp],[16,inp],[127,inp],[128,inp],[512,inp]]:
                        if s[0] > batch:
                            continue
                        print("- ",s)
                        tin = torch.randn(s)
                        tin.requires_grad = True
                        tout = final_op(tin)
                        dtout = torch.randn(tout.shape)
                        tout.backward(dtout,retain_graph=True)
                        case = dict(in_shapes = [ list(tin.shape)] ,out_shapes = [list(tout.shape)])
                        if np.prod(s) < 50 and test.get('random_params',False) == False:
                            case["in_tensors"] = [tin.reshape((-1,)).tolist()]
                            case["out_tensors"] = [tout.reshape((-1,)).tolist()]
                            case["out_diffs"] = [dtout.reshape((-1,)).tolist()]
                            case["in_diffs"] = [tin.grad.reshape((-1)).tolist()]
                            case["params_diffs"] = [ p.grad.reshape((-1,)).tolist() for p in params ]
                        else:
                            case["use_cpu_reference"]=True
                        for p in params:
                            p.grad*=0
                        cases.append(case)
    return report

def _at(x,n):
    if isinstance(x,(tuple,list)):
        return x[n]
    return x

def make_conv2d_gemm():
    return make_conv2d(algo="gemm")

def make_conv2d_win():
    return make_conv2d(algo="winograd")

def make_conv2d_dsc():
    return make_conv2d(algo="depthwise_separable")

def make_conv2d(algo=None):
    report = {
        "operator" : "Convolution2D",
        "tests" : []
    }
    tests = report["tests"]
    for kernel,pad,stride,dilate,cin,cout,bias,groups,relu,batch_limit in \
        [ 
            (5, 2, 1, 1,98,128,True,1,True,1), 
            (3, 1, 1, 1, 1, 1,False,1,False,64), 
            (3, 1, 1, 1, 2, 1,False,1,False,64), 
            (3, 1, 1, 1, 1, 2,False,1,False,64), 
            (3, 1, 1, 1, 3, 8,False,1,False,64), 
            (3, 1, 1, 1, 128, 64,False,1,False,64), 
            (3, 1, 1, 1, 3, 8,True, 1, True,64), 
            (3, 1, 1, 1, 3, 8,True, 1, True,64), 
            (1, 0, 1, 1, 3, 5,False,1,False,64),
            (1, 0, 1, 1, 48, 1152,False,1,False,64),
            (1, 0, 1, 1, 48, 1152,False,2,False,64),
            ([1,2], 0, 1, 1, 48, 1152,False,1,False,64),
            ([1,2], 0, 1, 1, 48, 1152,False,2,False,64),
            (1, 0, 2, 1, 3, 5,False,1,False,64),
            (3, 1, 1, 1, 4, 8,True,2,False,64), 
            (3, 1, 2, 1, 6, 8,True,2,False,64), 
            (3, 1, 1, 1, 8, 8,True,8,False,64), 
            ([1,7],[0,3],[1,1],[1,1],192,160,True,1,False,64),
            ([7,1],[3,0],[1,1],[1,1],128,192,True,1,False,64),
            ([3,5], [1,2], [2,3], [3,2], 3, 8,False,1,False,64),
            (5, 2, 1, 1, 16,32,True,1,False,64),
            (11,2, 4, 1, 16,32,True,1,False,64),
        ]:

        convop = torch.nn.Conv2d(cin,cout,kernel,stride=stride,padding=pad,dilation=dilate,groups=groups, bias=bias)
        params = list(convop.parameters())
        if relu:
            op = lambda x:(torch.nn.ReLU())(convop(x))
        else:
            op = convop
        cases=[]
        max_dim = 256 if cin * cout < 50000 else 10
        tin = torch.zeros(batch_limit,cin,max_dim,max_dim)
        tout = op(tin)
        test = {
            "init" : "small_frac",
            "train" : True,
            "options" : {
                "kernel": kernel,
                "pad" : pad,
                "stride" : stride,
                "dilate" : dilate,
                "bias" : bias,
                "channels_in" : cin,
                "channels_out" : cout,
                "groups" : groups
            },
            "setup_tensors" : [ { "shape" : list(tin.shape) } ],
            "output_tensors": [ { "shape" : list(tout.shape) } ],
            "param_specs":  [ { "shape" : list(p.shape) } for p in params ],
            "cases": cases
        }
        if algo is not None:
            test['options'].update(
                {
                    "fwd_algo" : algo,
                    "bwd_data_algo" : algo,
                    "bwd_filter_algo" : algo

                })

        if np.prod(params[0].shape) < 1000:
            pred_param=True
            test['param_tensors'] = [ p.reshape((-1,)).tolist() for p in params ]
        else:
            pred_param=False
            test['random_params'] = True
        if relu:
            test["options"]["activation"] = "relu"
        print(test["options"],"predefined params",pred_param)
        tests.append(test)
        for s in [[1,cin,2,2],[1,cin,7,7],[1,cin,8,8],[1,cin,4,4],[3,cin,4,4],[2,cin,7,7],[2,cin,10,5],[2,cin,10,10],[2,cin,19,19],[2,cin,20,20],[2,cin,32,32],
                  [64,cin,64,64],[53,cin,100,100]]:
            if s[0] > batch_limit:
                continue
            lkh,lkw = _at(kernel,0)*_at(dilate,0), _at(kernel,1)*_at(dilate,1)
            if s[2] + _at(pad,0) < lkh or s[3] + _at(pad,1) < lkw:
                continue
            if s[2] > max_dim or s[3] > max_dim:
                continue
            tin = torch.randn(s)
            tin.requires_grad = True
            tout = op(tin)
            dtout = torch.randn(tout.shape)
            tout.backward(dtout,retain_graph=True)
            case = dict(in_shapes = [ list(tin.shape)] ,out_shapes = [list(tout.shape)])
            if np.prod(s) < 5000 and pred_param:
                case["in_tensors"] = [tin.reshape((-1,)).tolist()]
                case["out_tensors"] = [tout.reshape((-1,)).tolist()]
                case["out_diffs"] = [dtout.reshape((-1,)).tolist()]
                case["in_diffs"] = [tin.grad.reshape((-1)).tolist()]
                case["params_diffs"] = [ p.grad.reshape((-1,)).tolist() for p in params ]
                print("- ",s,"predefined")
            else:
                case["use_cpu_reference"]=True
                print("- ",s,"cpu as ref")
            for p in params:
                p.grad*=0
            cases.append(case)
    return report

def make_tr_conv2d_gemm():
    return make_tr_conv2d(algo="gemm")

def make_tr_conv2d_win():
    return make_tr_conv2d(algo="winograd")

def make_tr_conv2d_dsc():
    return make_tr_conv2d(algo="depthwise_separable")


def make_tr_conv2d(algo=None):
    report = {
        "operator" : "TransposedConvolution2D",
        "tests" : []
    }
    tests = report["tests"]
    dev='cuda' #
    # There is but in PT for stride==2 https://github.com/pytorch/pytorch/issues/63314 so using gpu
    for kernel,pad,stride,dilate,cout,cin,bias,groups,relu,opad in \
        [ 
            (1, 0, 2, 1, 1, 1,False,1,False,0), 
            (1, 0, 2, 1, 1, 1,False,1,False,1), 
            (3, 1, 1, 1, 1, 1,False,1,False,0), 
            (3, 1, 1, 1, 2, 1,False,1,False,0), 
            (3, 1, 1, 1, 1, 2,False,1,False,0), 
            (3, 1, 1, 1, 3, 8,False,1,False,0), 
            (3, 1, 1, 1, 3, 8,False,1,False,0), 
            (3, 1, 1, 1, 128, 64,False,1,False,0), 
            (3, 1, 1, 1, 3, 8,True, 1, True,0), 
            (3, 1, 1, 1, 3, 8,True, 1, True,0), 
            (1, 0, 1, 1, 3, 5,False,1,False,0),
            (1, 0, 2, 1, 3, 5,False,1,False,0),
            (3, 1, 1, 1, 4, 8,True,2,False,0), 
            (3, 1, 2, 1, 6, 8,True,2,False,0), 
            (3, 1, 2, 1, 6, 8,True,2,False,1), 
            (3, 1, 1, 1, 8, 8,True,8,False,0), 
            ([3,5], [1,2], [2,3], [3,2], 3, 8,False,1,False,0),
            ([3,5], [1,2], [2,3], [3,2], 3, 8,False,1,False,[0,1]),
            (5, 2, 1, 1, 16,32,True,1,False,0),
            (11,2, 4, 1, 16,32,True,1,False,0),
            (11,2, 4, 1, 16,32,True,1,False,1),
        ]:

        convop = torch.nn.ConvTranspose2d(cin,cout,kernel,stride=stride,padding=pad,output_padding=opad,dilation=dilate,groups=groups, bias=bias).to(dev)
        params = list(convop.parameters())
        if relu:
            op = lambda x:(torch.nn.ReLU())(convop(x))
        else:
            op = convop
        cases=[]
        tin = torch.randn(64,cin,256,256).to(dev)
        tout = op(tin)
        test = {
            "init" : "small_frac",
            "train" : True,
            "options" : {
                "kernel": kernel,
                "pad" : pad,
                "output_pad": opad,
                "stride" : stride,
                "dilate" : dilate,
                "bias" : bias,
                "channels_in" : cin,
                "channels_out" : cout,
                "groups" : groups
            },
            "setup_tensors" : [ { "shape" : list(tin.shape) } ],
            "output_tensors": [ { "shape" : list(tout.shape) } ],
            "param_specs":  [ { "shape" : list(p.shape) } for p in params ],
            "cases": cases
        }
        if algo is not None:
            test['options'].update(
                {
                    "fwd_algo" : algo,
                    "bwd_data_algo" : algo,
                    "bwd_filter_algo" : algo

                })

        if np.prod(params[0].shape) < 1000:
            pred_param=True
            test['param_tensors'] = [ p.reshape((-1,)).tolist() for p in params ]
        else:
            pred_param=False
            test['random_params'] = True
        if relu:
            test["options"]["activation"] = "relu"
        print(test["options"],"predefined params",pred_param)
        tests.append(test)
        for sout in [[1,cout,2,2],[1,cout,7,7],[1,cout,4,4],[3,cout,4,4],[2,cout,7,7],[2,cout,10,5],[2,cout,10,10],[2,cout,19,19],[2,cout,20,20]]:
            ihw = [0,0]
            ohw = sout[2:]
            for d in range(2):
                ihw[d] = (ohw[d] - 1) * _at(stride,d) - 2 * _at(pad,d) + _at(dilate,d) * (_at(kernel,d) - 1) + _at(opad,d) + 1;

            valid_pad = True
            for d in range(2):
                ref = (ihw[d] + 2 * _at(pad,d) - _at(dilate,d)* (_at(kernel,d) - 1) -1) // _at(stride,d) + 1
                if ref != ohw[d]:
                    valid_pad = False
            if not valid_pad:
                assert opad > 0
                print("  Can't use output padding")
                continue
            s = [sout[0],cin,ihw[0],ihw[1]]
            lkh,lkw = _at(kernel,0)*_at(dilate,0), _at(kernel,1)*_at(dilate,1)
            if s[2] + _at(pad,0) < lkh or s[3] + _at(pad,1) < lkw:
                continue
            tin = torch.randn(s).to(dev)
            tin.requires_grad = True
            tout = op(tin)
            dtout = torch.randn(tout.shape).to(dev)
            tout.backward(dtout,retain_graph=True)
            case = dict(in_shapes = [ list(tin.shape)] ,out_shapes = [list(tout.shape)])
            if np.prod(s) < 5000 and pred_param:
                case["in_tensors"] = [tin.reshape((-1,)).tolist()]
                case["out_tensors"] = [tout.reshape((-1,)).tolist()]
                case["out_diffs"] = [dtout.reshape((-1,)).tolist()]
                case["in_diffs"] = [tin.grad.reshape((-1)).tolist()]
                case["params_diffs"] = [ p.grad.reshape((-1,)).tolist() for p in params ]
                print("- ",s,"predefined")
            else:
                case["use_cpu_reference"]=True
                print("- ",s,"cpu as ref")
            for p in params:
                p.grad*=0
            cases.append(case)
    return report






def gen(name,func): 
    torch.random.manual_seed(123)
    save_report((name,func()))

if __name__ == "__main__":
    cases  = { 
        "softmax" : make_softmax,
        "log_softmax" : make_log_softmax,
        "softmax_loss" : make_softmax_loss,
        "nll_loss" : make_nll_loss,
        "mse_loss" : make_mse_loss,
        "elementwise"  : make_eltwise,
        "pooling2d" : make_pooling2d,
        "global_pooling" : make_global_pooling,
        "inner_product" : make_inner_product,
        "conv2d" : make_conv2d,
        "conv2d_gemm" : make_conv2d_gemm,
        "conv2d_win" : make_conv2d_win,
        "conv2d_dsc" : make_conv2d_dsc,
        "tr_conv2d" : make_tr_conv2d,
        "tr_conv2d_gemm" : make_tr_conv2d_gemm,
        "tr_conv2d_win" : make_tr_conv2d_win,
        "tr_conv2d_dsc" : make_tr_conv2d_dsc,
        "activation" : make_activation,
        "batchnorm" : make_batchnorm,
        "concat" : make_concat,
        "slice" : make_slice,
        'threshold' : make_threshold,
        'hardtanh': make_hardtanh,
        'abs': make_abs,
        'reduction': make_reduction,
        'param': make_param,
        'interpolation': make_interpolation,
    }
    parse = argparse.ArgumentParser()
    parse.add_argument("--case",default="all",help="select case - one of " + ", ".join(list(cases) + ['all']))
    args = parse.parse_args()
    if args.case  == 'all':
        for case in cases:
            print("Generating ",case)
            gen(case,cases[case])
    else:
        gen(args.case,cases[args.case])
