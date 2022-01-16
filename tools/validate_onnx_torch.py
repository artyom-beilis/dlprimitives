import json
import os
from PIL import Image
import argparse
import time
import numpy as np
import dlprim as dp
import sys
import csv

def _prof_summary(report):
    sums=dict()
    counts=dict()
    summary=[]
    for line in [v for v in report.split('\n') if v]:
       row = [v for v in line.split(' ') if v]
       name=row[0]
       val=float(row[1])
       new_val = sums.get(name,0) + val
       new_cnt =counts.get(name,0) + 1
       sums[name ] = new_val
       counts[name] = new_cnt

    for name in sums:
        summary.append((name,sums[name],counts[name]))

    summary.sort(key = lambda x:x[1])
    print("Summary:")
    print("------")
    for r in summary:
        print("%10.5f %5d %s" % ( r[1],r[2],r[0]))
    print("------")



def export_torch_model(model,batch,path,opset = None):
    import torch
    inp = torch.randn(batch,3,224,224)
    model.eval()
    torch.onnx.export(model,inp,path,input_names = ["data"],output_names=["prob"],opset_version=opset)



class TorchModel(object):
    def __init__(self,model):
        self.model = model
    
    def eval(self,batch):
        import torch
        with torch.no_grad():
            self.model.eval();
            t = torch.from_numpy(batch)
            r = self.model(t)
            return r.detach().numpy()

class ONNXRTMode(object):
    def __init__(self,onnx_path):
        import onnxruntime as rt
        self.sess = rt.InferenceSession(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name

    def eval(self,batch):
        pred_onx = self.sess.run(None, {self.input_name: batch})[0]
        return pred_onx

class DLPrimModel(object):
    def __init__(self,onnx_path,device):
        onnx_model=dp.ONNXModel();
        onnx_model.load(onnx_path)
        self.ctx = dp.Context(device)
        self.net = dp.Net(self.ctx)
        self.net.mode = dp.PREDICT
        with open(onnx_path.replace('.onnx','.json'),'w') as f:
            f.write(onnx_model.network)
            f.write('\n')
        self.net.load_model(onnx_model)
        self.net.save_parameters(onnx_path.replace('.onnx','.dlp'))

    def eval(self,batch):
        data = self.net.tensor(self.net.input_names[0])
        data.reshape(dp.Shape(*batch.shape))
        self.net.reshape()
        prob = self.net.tensor(self.net.output_names[0])
        prob_cpu = np.zeros((prob.shape[0],prob.shape[1]),dtype=np.float32)
        q = self.ctx.make_execution_context(0)
        data.to_device(batch,q)
        self.net.forward(q)
        prob.to_host(prob_cpu,q)
        q.finish()
        return prob_cpu

class TFModel(object):
    def __init__(self,model):
        self.model = model
    
    def eval(self,batch):
        return self.model.predict_on_batch(batch)

def export_tf_model(model,batch,path,opset = None):
    import tf2onnx
    import tensorflow as tf
    onnx,_ = tf2onnx.convert.from_keras(model,input_signature=[tf.TensorSpec([batch,3,224,224], dtype=tf.dtypes.float32)],opset=opset)
    with open(path,'wb') as f:
        f.write(onnx.SerializeToString())


def predict_on_images(model,images,config):
    tw = 224
    th = 224
    mean = config['mean']
    std = config['std']
    classes = config['class_names']
    csv = []
    image = np.zeros((len(images),3,th,tw),dtype=np.float32)
    for i,path in enumerate(images):
        img = Image.open(path)
        npimg = np.array(img).astype(np.float32) * (1.0 / 255)
        h = npimg.shape[0]
        w = npimg.shape[1]
        assert h>=th
        assert w>=tw
        assert npimg.shape[2] == 3
        fact = 1.0 / np.array(std)
        off  = -np.array(mean) * fact
        dr = (h - th) // 2
        dc = (w - tw) // 2
        for k in range(3):
            image[i,k,:,:] = npimg[dr:dr+th,dc:dc+tw,k] * fact[k] + off[k]
    res = model.eval(image)
    for i in range(len(images)):
        index = np.argmax(res[i]).item()
        csv.append([images[i],str(index),classes[index]] + ['%8.6f' % v for v in res[i].tolist()])
    with open('report.csv','w') as f:
        for row in csv:
            line = ','.join(row) + '\n'
            f.write(line)
            sys.stdout.write(','.join(row[0:10] + ['...']) + '\n')
    return res


def get_config(fw,model):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(base_path + '/examples/cpp/imagenet_predict_config.json','r') as f:
        cfg = json.load(f)
    if fw == 'tf':
        if model.startswith('efficientnet'):
            cfg['mean'] = [0,0,0];
            cfg['std'] = [1/255.0,1/255.0,1/255.0]
        else:
            cfg['mean'] = [0.5,0.5,0.5]
            cfg['std'] =[0.5,0.5,0.5]

    return cfg

def get_tf_model(model_name):
    import tensorflow as tf
    tf.keras.backend.set_image_data_format('channels_first')
    if model_name == 'resnet50v2':
        return tf.keras.applications.ResNet50V2(classifier_activation=None)
    elif model_name == 'resnet50':
        return tf.keras.applications.ResNet50(classifier_activation=None)
    elif model_name == 'densenet121':
        return tf.keras.applications.DenseNet121()
    #elif model_name == 'nasnetmobile': channel last
    #    return tf.keras.applications.NASNetMobile()
    elif model_name == 'mobilenet':
        return tf.keras.applications.MobileNet(classifier_activation=None)
    elif model_name == 'mobilenet_v2':
        return tf.keras.applications.MobileNetV2(classifier_activation=None)
    elif model_name == 'vgg16':
        return tf.keras.applications.VGG16(classifier_activation=None)
    elif model_name == 'efficientnetb0':
        return tf.keras.applications.efficientnet.EfficientNetB0(classifier_activation=None)
    raise Exception("Invalid name " + model_name)

class MXModel(object):
    def __init__(self,model):
        self.model = model

    def eval(self,batch):
        import mxnet as mx
        mx_batch = mx.nd.array(batch)
        mx_res = self.model(mx_batch)
        np_res = mx_res.asnumpy()
        return np_res

def get_mx_model_and_export(name,batch,onnx_path):
    import mxnet as mx
    model = mx.gluon.model_zoo.vision.get_model(name,pretrained=True) 
    model.hybridize()
    model(mx.nd.array(np.random.randn(batch,3,224,224).astype(np.float32)))
    model.export('mx_model')
    mx.onnx.export_model('mx_model-symbol.json','mx_model-0000.params',in_shapes=[(batch,3,224,224)],in_types=[np.float32],onnx_file_path=onnx_path)
    return MXModel(model)


def main(args):
    onnx_path = args.model + ".onnx"
    config = get_config(args.fw,args.model)
    if args.fw == 'torch':
        import torch
        import torchvision
        m = getattr(torchvision.models,args.model)(pretrained = True)
        export_torch_model(m,args.batch,onnx_path,args.onnx_opset)
        src_model = TorchModel(m)
    elif args.fw == 'tf':
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        m = get_tf_model(args.model)
        export_tf_model(m,args.batch,onnx_path,args.onnx_opset)
        src_model = TFModel(m)
    elif args.fw == 'mx':
        src_model = get_mx_model_and_export(args.model,args.batch,onnx_path)
    else:
        raise Exception("Invalid framework " + args.fw)
    print("Framework:",args.fw)
    ref = predict_on_images(src_model,args.images,config)
    src_model = None
    if args.onnx_rt:
        print("ONNX Runtime")
        onx = predict_on_images(ONNXRTMode(onnx_path),args.images,config)
        err = np.max(np.abs(ref - onx))
        print("ONNX Runtime Error:",err)
    print("DLPrimitives")
    act = predict_on_images(DLPrimModel(onnx_path,args.device),args.images,config)
    err = np.max(np.abs(ref - act))
    print("Error:",err)
    if err > 1e4:
        print("Failed")
        sys.exit(1)

if __name__ == '__main__': 
    p = argparse.ArgumentParser()
    p.add_argument('--model',default='alexnet')
    p.add_argument('--fw',default='torch')
    p.add_argument('--device',default='0:0')
    p.add_argument('--onnx-rt',action='store_true',default=False)
    p.add_argument('--onnx-opset',default=9,type=int)
    p.add_argument('--batch',default=4,type=int)
    p.add_argument('images',nargs='*')
    r = p.parse_args()
    main(r)
