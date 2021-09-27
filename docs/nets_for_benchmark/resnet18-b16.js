{
    "inputs": [
        {
            "shape": [
                16,
                3,
                224,
                224
            ],
            "name": "data"
        }
    ],
    "outputs": [
        "loss"
    ],
    "operators": [
        {
            "name": "Conv_0",
            "type": "Convolution2D",
            "inputs": [
                "data"
            ],
            "outputs": [
                "123"
            ],
            "params": [
                "conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    7,
                    7
                ],
                "stride": [
                    2,
                    2
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    3,
                    3
                ],
                "channels_out": 64,
                "channels_in": 3
            }
        },
        {
            "name": "BatchNormalization_1",
            "type": "BatchNorm",
            "inputs": [
                "123"
            ],
            "outputs": [
                "129"
            ],
            "params": [
                "bn1.running_mean",
                "bn1.running_var",
                "bn1.weight",
                "bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_2",
            "type": "Activation",
            "inputs": [
                "129"
            ],
            "outputs": [
                "129"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "MaxPool_3",
            "type": "Pooling2D",
            "inputs": [
                "129"
            ],
            "outputs": [
                "130"
            ],
            "options": {
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    2,
                    2
                ],
                "pad": [
                    1,
                    1
                ],
                "count_include_pad": 0,
                "mode": "max"
            }
        },
        {
            "name": "Conv_4",
            "type": "Convolution2D",
            "inputs": [
                "130"
            ],
            "outputs": [
                "131"
            ],
            "params": [
                "layer1.0.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 64,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_5",
            "type": "BatchNorm",
            "inputs": [
                "131"
            ],
            "outputs": [
                "137"
            ],
            "params": [
                "layer1.0.bn1.running_mean",
                "layer1.0.bn1.running_var",
                "layer1.0.bn1.weight",
                "layer1.0.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_6",
            "type": "Activation",
            "inputs": [
                "137"
            ],
            "outputs": [
                "137"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_7",
            "type": "Convolution2D",
            "inputs": [
                "137"
            ],
            "outputs": [
                "138"
            ],
            "params": [
                "layer1.0.conv2.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 64,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_8",
            "type": "BatchNorm",
            "inputs": [
                "138"
            ],
            "outputs": [
                "139"
            ],
            "params": [
                "layer1.0.bn2.running_mean",
                "layer1.0.bn2.running_var",
                "layer1.0.bn2.weight",
                "layer1.0.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_9",
            "type": "Elementwise",
            "inputs": [
                "139",
                "130"
            ],
            "outputs": [
                "145"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_11",
            "type": "Convolution2D",
            "inputs": [
                "145"
            ],
            "outputs": [
                "146"
            ],
            "params": [
                "layer1.1.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 64,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_12",
            "type": "BatchNorm",
            "inputs": [
                "146"
            ],
            "outputs": [
                "152"
            ],
            "params": [
                "layer1.1.bn1.running_mean",
                "layer1.1.bn1.running_var",
                "layer1.1.bn1.weight",
                "layer1.1.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_13",
            "type": "Activation",
            "inputs": [
                "152"
            ],
            "outputs": [
                "152"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_14",
            "type": "Convolution2D",
            "inputs": [
                "152"
            ],
            "outputs": [
                "153"
            ],
            "params": [
                "layer1.1.conv2.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 64,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_15",
            "type": "BatchNorm",
            "inputs": [
                "153"
            ],
            "outputs": [
                "154"
            ],
            "params": [
                "layer1.1.bn2.running_mean",
                "layer1.1.bn2.running_var",
                "layer1.1.bn2.weight",
                "layer1.1.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_16",
            "type": "Elementwise",
            "inputs": [
                "154",
                "145"
            ],
            "outputs": [
                "160"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_18",
            "type": "Convolution2D",
            "inputs": [
                "160"
            ],
            "outputs": [
                "161"
            ],
            "params": [
                "layer2.0.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    2,
                    2
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 128,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_19",
            "type": "BatchNorm",
            "inputs": [
                "161"
            ],
            "outputs": [
                "167"
            ],
            "params": [
                "layer2.0.bn1.running_mean",
                "layer2.0.bn1.running_var",
                "layer2.0.bn1.weight",
                "layer2.0.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_20",
            "type": "Activation",
            "inputs": [
                "167"
            ],
            "outputs": [
                "167"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_21",
            "type": "Convolution2D",
            "inputs": [
                "167"
            ],
            "outputs": [
                "168"
            ],
            "params": [
                "layer2.0.conv2.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 128,
                "channels_in": 128
            }
        },
        {
            "name": "BatchNormalization_22",
            "type": "BatchNorm",
            "inputs": [
                "168"
            ],
            "outputs": [
                "169"
            ],
            "params": [
                "layer2.0.bn2.running_mean",
                "layer2.0.bn2.running_var",
                "layer2.0.bn2.weight",
                "layer2.0.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Conv_23",
            "type": "Convolution2D",
            "inputs": [
                "160"
            ],
            "outputs": [
                "174"
            ],
            "params": [
                "layer2.0.downsample.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
                ],
                "stride": [
                    2,
                    2
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    0,
                    0
                ],
                "channels_out": 128,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_24",
            "type": "BatchNorm",
            "inputs": [
                "174"
            ],
            "outputs": [
                "175"
            ],
            "params": [
                "layer2.0.downsample.1.running_mean",
                "layer2.0.downsample.1.running_var",
                "layer2.0.downsample.1.weight",
                "layer2.0.downsample.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_25",
            "type": "Elementwise",
            "inputs": [
                "169",
                "175"
            ],
            "outputs": [
                "181"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_27",
            "type": "Convolution2D",
            "inputs": [
                "181"
            ],
            "outputs": [
                "182"
            ],
            "params": [
                "layer2.1.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 128,
                "channels_in": 128
            }
        },
        {
            "name": "BatchNormalization_28",
            "type": "BatchNorm",
            "inputs": [
                "182"
            ],
            "outputs": [
                "188"
            ],
            "params": [
                "layer2.1.bn1.running_mean",
                "layer2.1.bn1.running_var",
                "layer2.1.bn1.weight",
                "layer2.1.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_29",
            "type": "Activation",
            "inputs": [
                "188"
            ],
            "outputs": [
                "188"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_30",
            "type": "Convolution2D",
            "inputs": [
                "188"
            ],
            "outputs": [
                "189"
            ],
            "params": [
                "layer2.1.conv2.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 128,
                "channels_in": 128
            }
        },
        {
            "name": "BatchNormalization_31",
            "type": "BatchNorm",
            "inputs": [
                "189"
            ],
            "outputs": [
                "190"
            ],
            "params": [
                "layer2.1.bn2.running_mean",
                "layer2.1.bn2.running_var",
                "layer2.1.bn2.weight",
                "layer2.1.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_32",
            "type": "Elementwise",
            "inputs": [
                "190",
                "181"
            ],
            "outputs": [
                "196"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_34",
            "type": "Convolution2D",
            "inputs": [
                "196"
            ],
            "outputs": [
                "197"
            ],
            "params": [
                "layer3.0.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    2,
                    2
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 256,
                "channels_in": 128
            }
        },
        {
            "name": "BatchNormalization_35",
            "type": "BatchNorm",
            "inputs": [
                "197"
            ],
            "outputs": [
                "203"
            ],
            "params": [
                "layer3.0.bn1.running_mean",
                "layer3.0.bn1.running_var",
                "layer3.0.bn1.weight",
                "layer3.0.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_36",
            "type": "Activation",
            "inputs": [
                "203"
            ],
            "outputs": [
                "203"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_37",
            "type": "Convolution2D",
            "inputs": [
                "203"
            ],
            "outputs": [
                "204"
            ],
            "params": [
                "layer3.0.conv2.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 256,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_38",
            "type": "BatchNorm",
            "inputs": [
                "204"
            ],
            "outputs": [
                "205"
            ],
            "params": [
                "layer3.0.bn2.running_mean",
                "layer3.0.bn2.running_var",
                "layer3.0.bn2.weight",
                "layer3.0.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Conv_39",
            "type": "Convolution2D",
            "inputs": [
                "196"
            ],
            "outputs": [
                "210"
            ],
            "params": [
                "layer3.0.downsample.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
                ],
                "stride": [
                    2,
                    2
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    0,
                    0
                ],
                "channels_out": 256,
                "channels_in": 128
            }
        },
        {
            "name": "BatchNormalization_40",
            "type": "BatchNorm",
            "inputs": [
                "210"
            ],
            "outputs": [
                "211"
            ],
            "params": [
                "layer3.0.downsample.1.running_mean",
                "layer3.0.downsample.1.running_var",
                "layer3.0.downsample.1.weight",
                "layer3.0.downsample.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_41",
            "type": "Elementwise",
            "inputs": [
                "205",
                "211"
            ],
            "outputs": [
                "217"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_43",
            "type": "Convolution2D",
            "inputs": [
                "217"
            ],
            "outputs": [
                "218"
            ],
            "params": [
                "layer3.1.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 256,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_44",
            "type": "BatchNorm",
            "inputs": [
                "218"
            ],
            "outputs": [
                "224"
            ],
            "params": [
                "layer3.1.bn1.running_mean",
                "layer3.1.bn1.running_var",
                "layer3.1.bn1.weight",
                "layer3.1.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_45",
            "type": "Activation",
            "inputs": [
                "224"
            ],
            "outputs": [
                "224"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_46",
            "type": "Convolution2D",
            "inputs": [
                "224"
            ],
            "outputs": [
                "225"
            ],
            "params": [
                "layer3.1.conv2.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 256,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_47",
            "type": "BatchNorm",
            "inputs": [
                "225"
            ],
            "outputs": [
                "226"
            ],
            "params": [
                "layer3.1.bn2.running_mean",
                "layer3.1.bn2.running_var",
                "layer3.1.bn2.weight",
                "layer3.1.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_48",
            "type": "Elementwise",
            "inputs": [
                "226",
                "217"
            ],
            "outputs": [
                "232"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_50",
            "type": "Convolution2D",
            "inputs": [
                "232"
            ],
            "outputs": [
                "233"
            ],
            "params": [
                "layer4.0.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    2,
                    2
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 512,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_51",
            "type": "BatchNorm",
            "inputs": [
                "233"
            ],
            "outputs": [
                "239"
            ],
            "params": [
                "layer4.0.bn1.running_mean",
                "layer4.0.bn1.running_var",
                "layer4.0.bn1.weight",
                "layer4.0.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_52",
            "type": "Activation",
            "inputs": [
                "239"
            ],
            "outputs": [
                "239"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_53",
            "type": "Convolution2D",
            "inputs": [
                "239"
            ],
            "outputs": [
                "240"
            ],
            "params": [
                "layer4.0.conv2.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 512,
                "channels_in": 512
            }
        },
        {
            "name": "BatchNormalization_54",
            "type": "BatchNorm",
            "inputs": [
                "240"
            ],
            "outputs": [
                "241"
            ],
            "params": [
                "layer4.0.bn2.running_mean",
                "layer4.0.bn2.running_var",
                "layer4.0.bn2.weight",
                "layer4.0.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Conv_55",
            "type": "Convolution2D",
            "inputs": [
                "232"
            ],
            "outputs": [
                "246"
            ],
            "params": [
                "layer4.0.downsample.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
                ],
                "stride": [
                    2,
                    2
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    0,
                    0
                ],
                "channels_out": 512,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_56",
            "type": "BatchNorm",
            "inputs": [
                "246"
            ],
            "outputs": [
                "247"
            ],
            "params": [
                "layer4.0.downsample.1.running_mean",
                "layer4.0.downsample.1.running_var",
                "layer4.0.downsample.1.weight",
                "layer4.0.downsample.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_57",
            "type": "Elementwise",
            "inputs": [
                "241",
                "247"
            ],
            "outputs": [
                "253"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_59",
            "type": "Convolution2D",
            "inputs": [
                "253"
            ],
            "outputs": [
                "254"
            ],
            "params": [
                "layer4.1.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 512,
                "channels_in": 512
            }
        },
        {
            "name": "BatchNormalization_60",
            "type": "BatchNorm",
            "inputs": [
                "254"
            ],
            "outputs": [
                "260"
            ],
            "params": [
                "layer4.1.bn1.running_mean",
                "layer4.1.bn1.running_var",
                "layer4.1.bn1.weight",
                "layer4.1.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_61",
            "type": "Activation",
            "inputs": [
                "260"
            ],
            "outputs": [
                "260"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_62",
            "type": "Convolution2D",
            "inputs": [
                "260"
            ],
            "outputs": [
                "261"
            ],
            "params": [
                "layer4.1.conv2.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    3,
                    3
                ],
                "stride": [
                    1,
                    1
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    1,
                    1
                ],
                "channels_out": 512,
                "channels_in": 512
            }
        },
        {
            "name": "BatchNormalization_63",
            "type": "BatchNorm",
            "inputs": [
                "261"
            ],
            "outputs": [
                "262"
            ],
            "params": [
                "layer4.1.bn2.running_mean",
                "layer4.1.bn2.running_var",
                "layer4.1.bn2.weight",
                "layer4.1.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_64",
            "type": "Elementwise",
            "inputs": [
                "262",
                "253"
            ],
            "outputs": [
                "268"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "GlobalAveragePool_66",
            "type": "GlobalPooling",
            "inputs": [
                "268"
            ],
            "outputs": [
                "270"
            ],
            "options": {
                "mode": "avg"
            }
        },
        {
            "name": "Gemm_68",
            "type": "InnerProduct",
            "inputs": [
                "270"
            ],
            "outputs": [
                "loss"
            ],
            "params": [
                "fc.weight",
                "fc.bias"
            ],
            "options": {
                "bias": true,
                "outputs": 1000,
                "inputs": 512
            }
        }
    ]
}