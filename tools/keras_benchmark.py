import os
os.environ['KERAS_BACKEND']='plaidml.keras.backend'
import time
import numpy as np
import sys
import json
import argparse


if 'KERAS_BACKEND' in os.environ:
    import keras as kr
    from keras.models import Sequential
    import keras.layers as layers
    from keras.utils import to_categorical
else:
    import tensorflow.keras as kr
    from tensorflow.keras.models import Sequential
    import tensorflow.keras.layers as layers
    from tensorflow.keras.utils import to_categorical

def make_model_from_dp(js):
    shape = tuple(js['inputs'][0]['shape'][1:])
    blobs={}
    input_layer = kr.Input(shape=shape,name = js['inputs'][0]['name'])
    blobs[js['inputs'][0]['name']] = input_layer
    for oper in js['operators']:
        tp = oper['type']
        inp= oper['inputs']
        out= oper['outputs']
        op = oper.get('options',{})
        if 'activation' in op:
            if op['activation'] == 'relu6':
                op['activation'] = 'relu' # workaround for missing layer - not important for benchmarks
        if tp == 'Convolution2D':
            pad=op['pad'][0]
            ker=op['kernel'][0]
            if pad * 2 + 1 != ker:
                padding = 'valid'
                prep = layers.ZeroPadding2D(padding = op['pad'])
            else:
                padding = 'same'
                prep = None
            activation = op.get('activation')
            if op['groups'] == op['channels_out'] and op['groups'] == op['channels_in']:
                l = layers.DepthwiseConv2D(
                                     kernel_size = op['kernel'],
                                     strides=op['stride'],
                                     padding = padding,
                                     activation = activation,
                                     use_bias = op['bias'])
            elif op['groups'] == 1:
                l = layers.Convolution2D(filters=op['channels_out'],
                                     kernel_size = op['kernel'],
                                     strides=op['stride'],
                                     dilation_rate = op['dilate'],
                                     padding = padding,
                                     activation = activation,
                                     use_bias = op['bias'])
            else:
                raise Exception('Unsupported groups count')
            if prep:
                blobs[out[0]] = l(prep(blobs[inp[0]]))
            else:
                blobs[out[0]] = l(blobs[inp[0]])
        elif tp == 'Pooling2D':
            pad=op['pad'][0]
            ker=op['kernel'][0]
            if pad * 2 + 1 != ker:
                padding = 'valid'
                prep = layers.ZeroPadding2D(padding = op['pad'])
            else:
                padding = 'same'
                prep = None
            if op['mode']=='max':
                l = layers.MaxPooling2D(pool_size=op['kernel'],strides=op['stride'],padding=padding)
            else:
                l = layers.AveragePooling2D(pool_size=op['kernel'],strides=op['stride'],padding=padding)
            if prep:
                blobs[out[0]] = l(prep(blobs[inp[0]]))
            else:
                blobs[out[0]] = l(blobs[inp[0]])
        elif tp == 'BatchNorm2D':
            affine = op.get('affine',True)
            l=layers.BatchNormalization(epsilon=op['eps'],momentum=(1-op['momentum']),scale=affine,center=affine)
            blobs[out[0]] = l(blobs[inp[0]])
        elif tp == 'Activation':
            blobs[out[0]] = layers.Activation(activation=op['activation'])(blobs[inp[0]])
        elif tp == 'Elementwise' and op['operation']=='sum':
            act = op.get('activation',None)
            if act:
                blobs[out[0]] =layers.Activation(activation=op['activation'])(layers.Add()([blobs[inp[0]],blobs[inp[1]]]))
            else:
                blobs[out[0]] =layers.Add()([blobs[inp[0]],blobs[inp[1]]])
        elif tp=='GlobalPooling' and op['mode']=='avg':
            blobs[out[0]] = layers.GlobalAveragePooling2D()(blobs[inp[0]])
        elif tp=='InnerProduct':
            if len(blobs[inp[0]].shape.dims)!=2:
                in_name = inp[0] + "_flatten"
                blobs[in_name] = layers.Flatten()(blobs[inp[0]])
            else:
                in_name = inp[0]
            blobs[out[0]] = layers.Dense(op['outputs'],use_bias=op['bias'],activation=op.get('activation'))(blobs[in_name])
        else:
            raise Exception("Unsupported layer %s/%s" % (tp,json.dumps(op)))
    return kr.Model(input_layer,blobs[js['outputs'][0]],name='amodel')


def make_model(path):
    with open(path,'r') as f:
        dp = json.load(f)
    model = make_model_from_dp(dp)
    print(model.summary())
    return model

def benchmark(model,batch_size,warm,iters):
    data = np.random.random((batch_size,3,224,224)).astype(np.float32) #.astype(np.half)
    lbls = np.random.randint(1000,size=(batch_size,))
    tgt = to_categorical(lbls,num_classes=1000)


    model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['accuracy'],
    )
    N=iters
    start = time.time()
    for k in range(-warm,N):
        if k == 0:
            start = time.time()
        p1 = time.time()
        model.train_on_batch(data,y=tgt)
        p2 = time.time()
        print("%5d %5.3fms" % (k,(p2-p1)*1000))
    tt = time.time() -  start
    train_time = tt * 1000 / N
    start = time.time()
    for k in range(-warm,N):
        if k == 0:
            start = time.time()
        p1 = time.time()
        model.test_on_batch(data,y=tgt)
        p2 = time.time()
        print("%5d %5.3fms" % (k,(p2-p1)*1000))
    tt = time.time() -  start
    test_time = tt * 1000 / N
    print("Total train %5.3f ms" % train_time)
    print("Total test %5.3f ms" % test_time)
    return tt


def run(path,batch,warm,iters):
    m = make_model(path)
    benchmark(m,batch,warm,iters)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model')
    p.add_argument('--batch',default=16,type=int)
    p.add_argument('--warm',default=5,type=int)
    p.add_argument('--iters',default=20,type=int)
    r = p.parse_args()
    run(r.model,r.batch,r.warm,r.iters)

