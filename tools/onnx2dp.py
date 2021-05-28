import sys
import onnx
import json
import argparse
import h5py

def get_inputs(model):
    params = dict()
    inputs = []
    for i in model.graph.initializer:
        name = i.name
        shape = i.dims
        params[name] = dict( shape = shape )

    for inp in model.graph.input:
        name = str(inp.name)
        if name in params:
            continue
        shape = [int(d.dim_value) for d in inp.type.tensor_type.shape.dim]
        inputs.append(dict(shape=shape,name=name))
    return inputs,params

def get_attrs(attrs):
    res = dict()
    for a in attrs:
        name = a.name
        if a.type == 1:
            res[name] = a.f
        elif a.type == 6:
            res[name] = list(a.floats)
        elif a.type == 2:
            res[name] = a.i
        elif a.type == 7:
            res[name] = list(a.ints)
        else:
            raise Exception("Unknow type ",a.type)
    return res

def get_pads(attrs):
    if 'pads' in attrs:
        pads = attrs['pads']
        assert len(pads) == 4
        assert pads[0] == pads[2] and pads[1] == pads[3]
        return list(pads[0:2])
    else:
        return [0,0]

def get_operators(model,inputs,params):
    operators = []
    actdic = {
        "Relu" : 'relu'
    }
    for n in model.graph.node:
        attrs = get_attrs(n.attribute)
        if n.op_type == 'Conv':
            pads = get_pads(attrs)
            op = dict(name = n.name,
                      type = 'Convolution2D',
                      inputs = [n.input[0]],
                      outputs = list(n.output),
                      params = list(n.input[1:]),
                      options = dict(
                        bias = len(n.input) > 2,
                        groups = attrs.get('group',1),
                        kernel = attrs['kernel_shape'],
                        stride = attrs.get('strides',1),
                        dilate = attrs.get('dilations',1),
                        pad = pads,
                        channels_out = params[n.input[1]]['shape'][0],
                        channels_in  = params[n.input[1]]['shape'][1]
                      )
                    )
            operators.append(op)
        elif n.op_type == 'Gemm':
            assert attrs['alpha'] == 1.0
            assert attrs['beta']  == 1.0
            assert attrs.get('transA',0) == 0
            assert attrs.get('transB',0) == 1
            op = dict(name = n.name,
                      type = 'InnerProduct',
                      inputs = [n.input[0]],
                      outputs = list(n.output),
                      params = list(n.input[1:]),
                      options = dict(
                        bias = len(n.input) > 2,
                        outputs = params[n.input[1]]['shape'][0],
                        inputs  = params[n.input[1]]['shape'][1]
                      )
                    )
            operators.append(op)
        elif n.op_type in actdic:
            opt_name = actdic[n.op_type]
            if  operators[-1]['options'].get('activation') is None \
                and operators[-1]['type'] in ('Convolution2D','InnerProduct','Elementwise') \
                and operators[-1]['outputs'][0] == n.input[0]:

                operators[-1]['options']['activation'] = opt_name
                operators[-1]['outputs'][0] = n.output[0]
            else:
                raise Exception("Need previous operator to add %s " % opt_name)
        elif n.op_type == 'Flatten':
            operators[-1]['outputs'][0] = n.output[0]
        elif n.op_type == 'MaxPool': 
            assert attrs.get('ceil_mode',0) == 0
            assert tuple(attrs.get('dilations',[1,1])) == (1,1)
            pads = get_pads(attrs)
            kern = attrs['kernel_shape']
            strides = attrs['strides']
            op = dict(name = n.name,
                      type = 'Pooling',
                      inputs = [n.input[0]],
                      outputs = list(n.output),
                      options = dict(
                        kernel = kern,
                        stride = strides,
                        pad = pads,
                        mode = 'max'
                      )
                    )
            operators.append(op)
        else:
            print("Unknown",n.op_type)
    return operators 
        
        
def main(o_path):
    model = onnx.load_model(o_path)
    inputs,params = get_inputs(model)
    operators = get_operators(model,inputs,params)

    dp = dict(
        inputs = inputs,
        operators = operators
    )
    with open('model_dp.json','w') as  f:
        json.dump(dp,f,indent=4)
    
    

if __name__ == "__main__":
    main(sys.argv[1])

