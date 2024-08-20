#!/bin/bash

fw="$1"
export PYTHONPATH=python

if [ "$fw" == "torch" ]
then
    for net in  alexnet vgg16 resnet18 resnext50_32x4d wide_resnet50_2 \
                efficientnet_b0 efficientnet_b4 regnet_y_400mf squeezenet1_0 \
                mobilenet_v2 densenet121
    do
        for opset in 9 11 13
        do
            echo "Net $net, opset $opset"
            python ../tools/validate_onnx_imagenet.py  --model $net --fw $fw --onnx-opset $opset ../tests/samples/*.ppm || exit 1
        done
    done
elif [ "$fw" == 'tf' ] 
then
    for net in resnet50 densenet121
    do
        for opset in 9 11
        do
            echo "Net $net, opset $opset"
            TF_CPP_MIN_LOG_LEVEL=3 python ../tools/validate_onnx_imagenet.py  --model $net --fw $fw --onnx-opset $opset ../tests/samples/*.ppm || exit 1
        done
    done
elif [ "$fw" == "mx" ]
then
    for net in vgg11_bn alexnet mobilenetv2_0.25 mobilenet0.25 densenet121 resnet18_v1 squeezenet1.0
    do
        echo "Net $net"
        python ../tools/validate_onnx_imagenet.py  --model $net --fw $fw ../tests/samples/*.ppm || exit 1
    done

else
    echo "Invalid environment $fw"
fi

