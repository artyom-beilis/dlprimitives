import torch
import json
import numpy as np
import argparse

def save_report(r):
    name,js = r
    with open('test_case_%s.json' %name,'w') as f:
        json.dump(js,f,indent=2)

def make_softmax():
    report = {
        "operator" : "SoftMax",
        "tests" : [
            {
                "options" : {},
                "setup_tensors" : [ {"shape":[10,50]} ],
                "output_tensors" : [ {"shape":[10,50]} ],
                "workspace": 0,
                "cases" : []
            }
        ]
    }
    cases = report["tests"][0]["cases"]
    sm = torch.nn.Softmax(dim=1)
    for b in [1,2,5,10,50,100,120]:
        for f in [1,5,31,32,33,63,64,65,100,1000,2000]:
            case = {
                "in_shapes"  : [[b,f]],
                "out_shapes" : [[b,f]],
            }
            if b <= 5 and f <= 32:
                inp = torch.randn(b,f) - 0.5
                out = sm(inp)
                case["in_tensors"] = [inp.reshape((-1,)).tolist()]
                case["out_tensors"] = [out.reshape((-1,)).tolist()]
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
                    for shape in [[5,2],[32,6,16,16],[128,256,16,16],[10023]]:
                        case = dict(in_shapes = [ shape, shape] ,out_shapes = [shape])
                        if np.prod(shape) < 100:
                            a = torch.randn(*shape) - 0.5
                            b = torch.randn(*shape) - 0.5
                            c = final_op(a,b)
                            case["in_tensors"] = [a.reshape((-1,)).tolist(),b.reshape((-1,)).tolist()]
                            case["out_tensors"] = [c.reshape((-1,)).tolist()]
                        else:
                            case["use_cpu_reference"]=True
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
          ([3,5],[1,2],[3,4]) ]:
          for tp,inc_pad,op in [ 
                ("max",False,torch.nn.MaxPool2d(krn,stride=stride,padding=pad)),
                ("avg",False,torch.nn.AvgPool2d(krn,stride=stride,padding=pad,count_include_pad=False)),
                ("avg",True, torch.nn.AvgPool2d(krn,stride=stride,padding=pad,count_include_pad=True)) 
            ]:
            cases=[]
            tin = torch.randn(4,16,32,32)
            tout = op(tin)
            test = {
                "options" : {
                    "mode": tp,
                    "kernel":krn,
                    "pad" : pad,
                    "stride" : stride,
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
                tin = torch.randn(s)
                tout = op(tin)
                case = dict(in_shapes = [ list(tin.shape)] ,out_shapes = [list(tout.shape)])
                if np.prod(s) < 200:
                    case["in_tensors"] = [tin.reshape((-1,)).tolist()]
                    case["out_tensors"] = [tout.reshape((-1,)).tolist()]
                else:
                    case["use_cpu_reference"]=True
                cases.append(case)
    return report
    
def make_inner_product():
    report = {
        "operator" : "InnerProduct",
        "tests" : []
    }
    tests = report["tests"]
    for inp,out in \
        [ (5,11),
          (1024,1024),
          (500,1000) ]:
          for bias in [True,False]:
            op = torch.nn.Linear(inp,out,bias=bias) 
            for act,mact in [ ("identity",lambda x:x), ("relu",torch.nn.ReLU()) ]:
                params = list(op.parameters())
                cases=[]
                tin = torch.randn(10,inp)
                tout = op(tin)
                test = {
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
                for s in [[2,inp],[8,inp],[127,inp],[128,inp]]:
                    print("- ",s)
                    tin = torch.randn(s)
                    tout = final_op(tin)
                    case = dict(in_shapes = [ list(tin.shape)] ,out_shapes = [list(tout.shape)])
                    if np.prod(s) < 50:
                        case["in_tensors"] = [tin.reshape((-1,)).tolist()]
                        case["out_tensors"] = [tout.reshape((-1,)).tolist()]
                    else:
                        case["use_cpu_reference"]=True
                    cases.append(case)
    return report




def gen(name,func): 
    torch.random.manual_seed(123)
    save_report((name,func()))

if __name__ == "__main__":
    cases  = { 
        "softmax" : make_softmax,
        "elementwise"  : make_eltwise,
        "pooling2d" : make_pooling2d,
        "inner_product" : make_inner_product
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
