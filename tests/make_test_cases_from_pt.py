import torch
import json
import numpy as np

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
    return 'softmax',report

def make_eltwise():
    report = {
        "operator" : "Elementwise",
        "tests" : []
    }
    tests = report["tests"]
    for op,mop in [ ("sum", lambda x,y:x+y),
                    ("prod",lambda x,y:x*y),
                    ("max" ,lambda x,y:torch.maximum(x,y)) ]:
        for c1 in [ -1.5, 1.0, 2.0 ]:
            for c2 in [ -3, 1, 5 ]:
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
    return 'elementwise',report


def gen(func): 
    torch.random.manual_seed(123)
    save_report(func())

if __name__ == "__main__":
    gen(make_softmax)
    gen(make_eltwise)
            
