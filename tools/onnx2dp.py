import sys
import onnx
import json
import argparse
import h5py
import numpy as np
import argparse

def get_inputs(model):
    params = dict()
    inputs = []
    outputs = []
    for i in model.graph.initializer:
        name = i.name
        shape = list(i.dims)
        print("Loading tensor %s/%s" % (name,shape))
        if i.data_type!= 1:
            raise Exception("Only floats are supported")
        if i.float_data:
            value = np.array(list(i.float_data))
        elif i.raw_data:
            value = np.frombuffer(i.raw_data,dtype=np.float32).copy()
        else:
            raise Exception("Can't read data: " + name)
        value = value.reshape(shape).astype(np.float32)
        params[name] = dict( shape = shape, value = value )

    for inp in model.graph.input:
        name = str(inp.name)
        if name in params:
            continue
        shape = [int(d.dim_value) for d in inp.type.tensor_type.shape.dim]
        inputs.append(dict(shape=shape,name=name))
    for out in model.graph.output:
        outputs.append(str(out.name))
    return inputs,outputs,params

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
    op_len = 0
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
        elif n.op_type in ('Flatten','Dropout'):
            assert operators[-1]['outputs'][0] == n.input[0]
            operators[-1]['outputs'][0] = n.output[0]
        elif n.op_type == 'Softmax':
            assert attrs['axis'] == 1
            op = dict(name = n.name,
                      type = 'SoftMax',
                      inputs = [n.input[0]],
                      outputs = list(n.output),
                    )
            operators.append(op)
        elif n.op_type == 'Add':
            op = dict(name = n.name,
                      type = 'Elementwise',
                      inputs = list(n.input),
                      outputs = list(n.output),
                      options = dict( operation = 'sum')
            )
            operators.append(op)
        elif n.op_type in ('GlobalAveragePool','GlobalMaxPool'):
            op = dict(name = n.name,
                      type = 'GlobalPooling',
                      inputs = [n.input[0]],
                      outputs = list(n.output),
                      options = dict(
                        mode = ('max' if n.op_type == 'GlobalMaxPool' else 'avg')
                      )
                    )
            operators.append(op)
             
        elif n.op_type == 'MaxPool' or n.op_type == 'AveragePool': 
            assert attrs.get('ceil_mode',0) == 0
            assert tuple(attrs.get('dilations',[1,1])) == (1,1)
            pads = get_pads(attrs)
            kern = attrs['kernel_shape']
            strides = attrs['strides']
            count_include_pad = attrs.get('count_include_pad',0)
            op = dict(name = n.name,
                      type = 'Pooling2D',
                      inputs = [n.input[0]],
                      outputs = list(n.output),
                      options = dict(
                        kernel = kern,
                        stride = strides,
                        pad = pads,
                        count_include_pad = count_include_pad,
                        mode = ('max' if n.op_type == 'MaxPool' else 'avg')
                      )
                    )
            operators.append(op)
        else:
            raise Exception("Unsupported operation: " + str(n.op_type) + " with attributes " + json.dumps(attrs));
        if op_len == len(operators):
            print("Skipped or modified last op from %s" % str(n.op_type))
        else:
            last = operators[-1]
            print("Adding operator %s from %s with options %s" % 
                (last['type'],n.op_type,json.dumps(last.get("options",{}))))
        op_len = len(operators)
    return operators 

def make_h5(file_name,params):
    h = h5py.File(file_name,'w')
    try:
        for name in params:
            shape = params[name]['shape']
            value = params[name]['value']
            ds = h.create_dataset(name,shape)
            ds[:] = value
    finally:
        h.close()
    
def convert_onnx_to_dlprim(o_path,js,h5):    
    model = onnx.load_model(o_path)
    inputs,outputs,params = get_inputs(model)
    print("Inputs",inputs)
    print("Outputs",outputs)
    operators = get_operators(model,inputs,params)

    dp = dict(
        inputs = inputs,
        outputs = outputs,
        operators = operators
    )
    print("Saving network")
    with open(js,'w') as  f:
        json.dump(dp,f,indent=4)
    print("Saving weights")
    make_h5(h5,params) 
    print("Done")


    
    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--js',default='model_dp.js')
    p.add_argument('--h5',default='model_dp.h5')
    r = p.parse_args()
    convert_onnx_to_dlprim(r.model,r.js,r.h5)

