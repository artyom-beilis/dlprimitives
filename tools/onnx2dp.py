###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
import sys
import struct
import onnx
import json
import os
import argparse
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
            print(i)
            print("Only floats are supported, given %d" % i.data_type)
            #raise Exception("Only floats are supported, given %d" % i.data_type)
            continue
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

def get_attrs(attrs,tp):
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
        elif a.type == 3:
            res[name] = str(a.s)
        else:
            print("Warning Unknow attibute type " + str(a) + "in operator" + tp)
            #raise Exception("Unknow attibute type " + str(a) + "in operator" + tp)
    return res

def get_pads(attrs):
    if 'pads' in attrs:
        pads = attrs['pads']
        assert len(pads) == 4
        #assert pads[0] == pads[2] and pads[1] == pads[3]
        return list(pads[0:2])
    else:
        return [0,0]

def get_operators(model,inputs,params):
    operators = []
    actdic = {
        "Relu" : 'relu',
        'Sigmoid' : 'sigmoid',
        "Clip" : 'relu6',
        "Pad" : "identity"
    }
    op_len = 0
    print("================")
    for n in model.graph.node:
        print(n.op_type,n.name)
    print("================")
    for n in model.graph.node:
        attrs = get_attrs(n.attribute,n.op_type)
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
                        channels_in  = params[n.input[1]]['shape'][1]*attrs.get('group',1)
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
        elif n.op_type == 'BatchNormalization':
            eps = attrs["epsilon"]
            momentum = 1.0 - attrs['momentum']
            op = dict(name = n.name,
                      type = 'BatchNorm',
                      inputs = [n.input[0]],
                      outputs = [n.output[0]],
                      params = list(n.input[3:5] + n.input[1:3]),
                      options = dict(
                        eps = eps,
                        momentum = momentum
                      )
                    )
            operators.append(op)
        elif n.op_type in actdic:
            opt_name = actdic[n.op_type]
            if opt_name == 'Clip' and (attrs.get('min')!=0  or attr.get('max')!=6):
                raise Exception("Clip expected to have min/max for Relu6")
            if  operators[-1]['options'].get('activation') is None \
                and operators[-1]['type'] in ('Convolution2D','InnerProduct','Elementwise') \
                and operators[-1]['outputs'][0] == n.input[0]:

                operators[-1]['options']['activation'] = opt_name
                operators[-1]['outputs'][0] = n.output[0]
            else:
                if operators[-1]['outputs'][0] == n.input[0]:
                    operators[-1]['outputs'][0] = n.output[0]
                    inp = n.output[0]
                else:
                    inp = n.input[0]
                op = dict(name = n.name,
                          type='Activation',
                          inputs = [inp],
                          outputs = list(n.output),
                          options = dict(activation=opt_name))
                operators.append(op)
        elif n.op_type in ('Flatten','Dropout','Reshape'):
            #assert operators[-1]['outputs'][0] == n.input[0]
            operators[-1]['outputs'][0] = n.output[0]
        elif n.op_type == 'Concat':
            op = dict(name = n.name,
                      type = 'Concat',
                      inputs = list(n.input),
                      outputs = list(n.output),
                      options = dict(dim = attrs['axis'])
                    )
            operators.append(op)
        elif n.op_type == 'Softmax':
            #assert attrs.get('axis',-1) in (1,-1)
            op = dict(name = n.name,
                      type = 'Softmax',
                      inputs = [n.input[0]],
                      outputs = list(n.output),
                    )
            operators.append(op)
        elif n.op_type == 'Mul':
            op = dict(name = n.name,
                      type = 'Elementwise',
                      inputs = list(n.input),
                      outputs = list(n.output),
                      options = dict( operation = 'prod')
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
            assert tuple(attrs.get('dilations',[1,1])) == (1,1)
            ceil_mode = bool(attrs.get('ceil_mode',0))
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
                        ceil_mode = ceil_mode,
                        count_include_pad = count_include_pad,
                        mode = ('max' if n.op_type == 'MaxPool' else 'avg')
                      )
                    )
            operators.append(op)
        else:
            print("Warning Unsupported operation: " + str(n.op_type) + " with attributes " + json.dumps(attrs));
            print("  Inputs",list(n.input)," Outputs",list(n.output))
            print("  Final network may not represent actual!!!")
        if op_len == len(operators):
            print("Skipped or modified last op from %s" % str(n.op_type))
        else:
            last = operators[-1]
            print("Adding operator %s from %s with options %s" % 
                (last['type'],n.op_type,json.dumps(last.get("options",{}))))
        op_len = len(operators)
    return operators 

def make_h5(file_name,params):
    import h5py
    h = h5py.File(file_name,'w')
    try:
        for name in params:
            shape = params[name]['shape']
            value = params[name]['value']
            ds = h.create_dataset(name,shape)
            ds[:] = value
    finally:
        h.close()

def make_dlp(file_name,params):
    with open(file_name,'wb') as f:
        start = 0
        tensors={}
        for name in params:
            shape = params[name]['shape']
            value = params[name]['value']
            length = value.nbytes
            tensors[name] = dict(shape=shape,dtype='float',start=start,size=length)
            start += length
        hjs = dict(tensors=tensors)
        h = json.dumps(hjs).encode();
        f.write(b'DLPW')
        f.write(struct.pack('!i',len(h)))
        f.write(h)
        for name in params:
            blob = params[name]['value'].tobytes()
            f.write(blob)
            start += len(blob)

    
def convert_onnx_to_dlprim(o_path,js,h5,dlp):    
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
    if dlp:
        make_dlp(dlp,params)
    if h5:
        make_h5(h5,params) 
    print("Done")


    
    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--h5')
    p.add_argument('--js',default='model_dp.js')
    p.add_argument('--dlp',default='model_dp.dlp')
    r = p.parse_args()
    convert_onnx_to_dlprim(r.model,r.js,r.h5,r.dlp)

