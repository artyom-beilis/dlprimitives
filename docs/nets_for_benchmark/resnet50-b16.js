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
                "321"
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
                "321"
            ],
            "outputs": [
                "327"
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
                "327"
            ],
            "outputs": [
                "327"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "MaxPool_3",
            "type": "Pooling2D",
            "inputs": [
                "327"
            ],
            "outputs": [
                "328"
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
                "328"
            ],
            "outputs": [
                "329"
            ],
            "params": [
                "layer1.0.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 64,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_5",
            "type": "BatchNorm",
            "inputs": [
                "329"
            ],
            "outputs": [
                "335"
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
                "335"
            ],
            "outputs": [
                "335"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_7",
            "type": "Convolution2D",
            "inputs": [
                "335"
            ],
            "outputs": [
                "336"
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
                "336"
            ],
            "outputs": [
                "342"
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
            "name": "Relu_9",
            "type": "Activation",
            "inputs": [
                "342"
            ],
            "outputs": [
                "342"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_10",
            "type": "Convolution2D",
            "inputs": [
                "342"
            ],
            "outputs": [
                "343"
            ],
            "params": [
                "layer1.0.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 256,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_11",
            "type": "BatchNorm",
            "inputs": [
                "343"
            ],
            "outputs": [
                "344"
            ],
            "params": [
                "layer1.0.bn3.running_mean",
                "layer1.0.bn3.running_var",
                "layer1.0.bn3.weight",
                "layer1.0.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Conv_12",
            "type": "Convolution2D",
            "inputs": [
                "328"
            ],
            "outputs": [
                "349"
            ],
            "params": [
                "layer1.0.downsample.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 256,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_13",
            "type": "BatchNorm",
            "inputs": [
                "349"
            ],
            "outputs": [
                "350"
            ],
            "params": [
                "layer1.0.downsample.1.running_mean",
                "layer1.0.downsample.1.running_var",
                "layer1.0.downsample.1.weight",
                "layer1.0.downsample.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_14",
            "type": "Elementwise",
            "inputs": [
                "344",
                "350"
            ],
            "outputs": [
                "356"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_16",
            "type": "Convolution2D",
            "inputs": [
                "356"
            ],
            "outputs": [
                "357"
            ],
            "params": [
                "layer1.1.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 64,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_17",
            "type": "BatchNorm",
            "inputs": [
                "357"
            ],
            "outputs": [
                "363"
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
            "name": "Relu_18",
            "type": "Activation",
            "inputs": [
                "363"
            ],
            "outputs": [
                "363"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_19",
            "type": "Convolution2D",
            "inputs": [
                "363"
            ],
            "outputs": [
                "364"
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
            "name": "BatchNormalization_20",
            "type": "BatchNorm",
            "inputs": [
                "364"
            ],
            "outputs": [
                "370"
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
            "name": "Relu_21",
            "type": "Activation",
            "inputs": [
                "370"
            ],
            "outputs": [
                "370"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_22",
            "type": "Convolution2D",
            "inputs": [
                "370"
            ],
            "outputs": [
                "371"
            ],
            "params": [
                "layer1.1.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 256,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_23",
            "type": "BatchNorm",
            "inputs": [
                "371"
            ],
            "outputs": [
                "372"
            ],
            "params": [
                "layer1.1.bn3.running_mean",
                "layer1.1.bn3.running_var",
                "layer1.1.bn3.weight",
                "layer1.1.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_24",
            "type": "Elementwise",
            "inputs": [
                "372",
                "356"
            ],
            "outputs": [
                "378"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_26",
            "type": "Convolution2D",
            "inputs": [
                "378"
            ],
            "outputs": [
                "379"
            ],
            "params": [
                "layer1.2.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 64,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_27",
            "type": "BatchNorm",
            "inputs": [
                "379"
            ],
            "outputs": [
                "385"
            ],
            "params": [
                "layer1.2.bn1.running_mean",
                "layer1.2.bn1.running_var",
                "layer1.2.bn1.weight",
                "layer1.2.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_28",
            "type": "Activation",
            "inputs": [
                "385"
            ],
            "outputs": [
                "385"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_29",
            "type": "Convolution2D",
            "inputs": [
                "385"
            ],
            "outputs": [
                "386"
            ],
            "params": [
                "layer1.2.conv2.weight"
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
            "name": "BatchNormalization_30",
            "type": "BatchNorm",
            "inputs": [
                "386"
            ],
            "outputs": [
                "392"
            ],
            "params": [
                "layer1.2.bn2.running_mean",
                "layer1.2.bn2.running_var",
                "layer1.2.bn2.weight",
                "layer1.2.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_31",
            "type": "Activation",
            "inputs": [
                "392"
            ],
            "outputs": [
                "392"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_32",
            "type": "Convolution2D",
            "inputs": [
                "392"
            ],
            "outputs": [
                "393"
            ],
            "params": [
                "layer1.2.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 256,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_33",
            "type": "BatchNorm",
            "inputs": [
                "393"
            ],
            "outputs": [
                "394"
            ],
            "params": [
                "layer1.2.bn3.running_mean",
                "layer1.2.bn3.running_var",
                "layer1.2.bn3.weight",
                "layer1.2.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_34",
            "type": "Elementwise",
            "inputs": [
                "394",
                "378"
            ],
            "outputs": [
                "400"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_36",
            "type": "Convolution2D",
            "inputs": [
                "400"
            ],
            "outputs": [
                "401"
            ],
            "params": [
                "layer2.0.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 128,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_37",
            "type": "BatchNorm",
            "inputs": [
                "401"
            ],
            "outputs": [
                "407"
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
            "name": "Relu_38",
            "type": "Activation",
            "inputs": [
                "407"
            ],
            "outputs": [
                "407"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_39",
            "type": "Convolution2D",
            "inputs": [
                "407"
            ],
            "outputs": [
                "408"
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
                "channels_in": 128
            }
        },
        {
            "name": "BatchNormalization_40",
            "type": "BatchNorm",
            "inputs": [
                "408"
            ],
            "outputs": [
                "414"
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
            "name": "Relu_41",
            "type": "Activation",
            "inputs": [
                "414"
            ],
            "outputs": [
                "414"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_42",
            "type": "Convolution2D",
            "inputs": [
                "414"
            ],
            "outputs": [
                "415"
            ],
            "params": [
                "layer2.0.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 512,
                "channels_in": 128
            }
        },
        {
            "name": "BatchNormalization_43",
            "type": "BatchNorm",
            "inputs": [
                "415"
            ],
            "outputs": [
                "416"
            ],
            "params": [
                "layer2.0.bn3.running_mean",
                "layer2.0.bn3.running_var",
                "layer2.0.bn3.weight",
                "layer2.0.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Conv_44",
            "type": "Convolution2D",
            "inputs": [
                "400"
            ],
            "outputs": [
                "421"
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
                "channels_out": 512,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_45",
            "type": "BatchNorm",
            "inputs": [
                "421"
            ],
            "outputs": [
                "422"
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
            "name": "Add_46",
            "type": "Elementwise",
            "inputs": [
                "416",
                "422"
            ],
            "outputs": [
                "428"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_48",
            "type": "Convolution2D",
            "inputs": [
                "428"
            ],
            "outputs": [
                "429"
            ],
            "params": [
                "layer2.1.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 128,
                "channels_in": 512
            }
        },
        {
            "name": "BatchNormalization_49",
            "type": "BatchNorm",
            "inputs": [
                "429"
            ],
            "outputs": [
                "435"
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
            "name": "Relu_50",
            "type": "Activation",
            "inputs": [
                "435"
            ],
            "outputs": [
                "435"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_51",
            "type": "Convolution2D",
            "inputs": [
                "435"
            ],
            "outputs": [
                "436"
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
            "name": "BatchNormalization_52",
            "type": "BatchNorm",
            "inputs": [
                "436"
            ],
            "outputs": [
                "442"
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
            "name": "Relu_53",
            "type": "Activation",
            "inputs": [
                "442"
            ],
            "outputs": [
                "442"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_54",
            "type": "Convolution2D",
            "inputs": [
                "442"
            ],
            "outputs": [
                "443"
            ],
            "params": [
                "layer2.1.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 512,
                "channels_in": 128
            }
        },
        {
            "name": "BatchNormalization_55",
            "type": "BatchNorm",
            "inputs": [
                "443"
            ],
            "outputs": [
                "444"
            ],
            "params": [
                "layer2.1.bn3.running_mean",
                "layer2.1.bn3.running_var",
                "layer2.1.bn3.weight",
                "layer2.1.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_56",
            "type": "Elementwise",
            "inputs": [
                "444",
                "428"
            ],
            "outputs": [
                "450"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_58",
            "type": "Convolution2D",
            "inputs": [
                "450"
            ],
            "outputs": [
                "451"
            ],
            "params": [
                "layer2.2.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 128,
                "channels_in": 512
            }
        },
        {
            "name": "BatchNormalization_59",
            "type": "BatchNorm",
            "inputs": [
                "451"
            ],
            "outputs": [
                "457"
            ],
            "params": [
                "layer2.2.bn1.running_mean",
                "layer2.2.bn1.running_var",
                "layer2.2.bn1.weight",
                "layer2.2.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_60",
            "type": "Activation",
            "inputs": [
                "457"
            ],
            "outputs": [
                "457"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_61",
            "type": "Convolution2D",
            "inputs": [
                "457"
            ],
            "outputs": [
                "458"
            ],
            "params": [
                "layer2.2.conv2.weight"
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
            "name": "BatchNormalization_62",
            "type": "BatchNorm",
            "inputs": [
                "458"
            ],
            "outputs": [
                "464"
            ],
            "params": [
                "layer2.2.bn2.running_mean",
                "layer2.2.bn2.running_var",
                "layer2.2.bn2.weight",
                "layer2.2.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_63",
            "type": "Activation",
            "inputs": [
                "464"
            ],
            "outputs": [
                "464"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_64",
            "type": "Convolution2D",
            "inputs": [
                "464"
            ],
            "outputs": [
                "465"
            ],
            "params": [
                "layer2.2.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 512,
                "channels_in": 128
            }
        },
        {
            "name": "BatchNormalization_65",
            "type": "BatchNorm",
            "inputs": [
                "465"
            ],
            "outputs": [
                "466"
            ],
            "params": [
                "layer2.2.bn3.running_mean",
                "layer2.2.bn3.running_var",
                "layer2.2.bn3.weight",
                "layer2.2.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_66",
            "type": "Elementwise",
            "inputs": [
                "466",
                "450"
            ],
            "outputs": [
                "472"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_68",
            "type": "Convolution2D",
            "inputs": [
                "472"
            ],
            "outputs": [
                "473"
            ],
            "params": [
                "layer2.3.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 128,
                "channels_in": 512
            }
        },
        {
            "name": "BatchNormalization_69",
            "type": "BatchNorm",
            "inputs": [
                "473"
            ],
            "outputs": [
                "479"
            ],
            "params": [
                "layer2.3.bn1.running_mean",
                "layer2.3.bn1.running_var",
                "layer2.3.bn1.weight",
                "layer2.3.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_70",
            "type": "Activation",
            "inputs": [
                "479"
            ],
            "outputs": [
                "479"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_71",
            "type": "Convolution2D",
            "inputs": [
                "479"
            ],
            "outputs": [
                "480"
            ],
            "params": [
                "layer2.3.conv2.weight"
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
            "name": "BatchNormalization_72",
            "type": "BatchNorm",
            "inputs": [
                "480"
            ],
            "outputs": [
                "486"
            ],
            "params": [
                "layer2.3.bn2.running_mean",
                "layer2.3.bn2.running_var",
                "layer2.3.bn2.weight",
                "layer2.3.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_73",
            "type": "Activation",
            "inputs": [
                "486"
            ],
            "outputs": [
                "486"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_74",
            "type": "Convolution2D",
            "inputs": [
                "486"
            ],
            "outputs": [
                "487"
            ],
            "params": [
                "layer2.3.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 512,
                "channels_in": 128
            }
        },
        {
            "name": "BatchNormalization_75",
            "type": "BatchNorm",
            "inputs": [
                "487"
            ],
            "outputs": [
                "488"
            ],
            "params": [
                "layer2.3.bn3.running_mean",
                "layer2.3.bn3.running_var",
                "layer2.3.bn3.weight",
                "layer2.3.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_76",
            "type": "Elementwise",
            "inputs": [
                "488",
                "472"
            ],
            "outputs": [
                "494"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_78",
            "type": "Convolution2D",
            "inputs": [
                "494"
            ],
            "outputs": [
                "495"
            ],
            "params": [
                "layer3.0.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 256,
                "channels_in": 512
            }
        },
        {
            "name": "BatchNormalization_79",
            "type": "BatchNorm",
            "inputs": [
                "495"
            ],
            "outputs": [
                "501"
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
            "name": "Relu_80",
            "type": "Activation",
            "inputs": [
                "501"
            ],
            "outputs": [
                "501"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_81",
            "type": "Convolution2D",
            "inputs": [
                "501"
            ],
            "outputs": [
                "502"
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
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_82",
            "type": "BatchNorm",
            "inputs": [
                "502"
            ],
            "outputs": [
                "508"
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
            "name": "Relu_83",
            "type": "Activation",
            "inputs": [
                "508"
            ],
            "outputs": [
                "508"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_84",
            "type": "Convolution2D",
            "inputs": [
                "508"
            ],
            "outputs": [
                "509"
            ],
            "params": [
                "layer3.0.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 1024,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_85",
            "type": "BatchNorm",
            "inputs": [
                "509"
            ],
            "outputs": [
                "510"
            ],
            "params": [
                "layer3.0.bn3.running_mean",
                "layer3.0.bn3.running_var",
                "layer3.0.bn3.weight",
                "layer3.0.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Conv_86",
            "type": "Convolution2D",
            "inputs": [
                "494"
            ],
            "outputs": [
                "515"
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
                "channels_out": 1024,
                "channels_in": 512
            }
        },
        {
            "name": "BatchNormalization_87",
            "type": "BatchNorm",
            "inputs": [
                "515"
            ],
            "outputs": [
                "516"
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
            "name": "Add_88",
            "type": "Elementwise",
            "inputs": [
                "510",
                "516"
            ],
            "outputs": [
                "522"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_90",
            "type": "Convolution2D",
            "inputs": [
                "522"
            ],
            "outputs": [
                "523"
            ],
            "params": [
                "layer3.1.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 256,
                "channels_in": 1024
            }
        },
        {
            "name": "BatchNormalization_91",
            "type": "BatchNorm",
            "inputs": [
                "523"
            ],
            "outputs": [
                "529"
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
            "name": "Relu_92",
            "type": "Activation",
            "inputs": [
                "529"
            ],
            "outputs": [
                "529"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_93",
            "type": "Convolution2D",
            "inputs": [
                "529"
            ],
            "outputs": [
                "530"
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
            "name": "BatchNormalization_94",
            "type": "BatchNorm",
            "inputs": [
                "530"
            ],
            "outputs": [
                "536"
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
            "name": "Relu_95",
            "type": "Activation",
            "inputs": [
                "536"
            ],
            "outputs": [
                "536"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_96",
            "type": "Convolution2D",
            "inputs": [
                "536"
            ],
            "outputs": [
                "537"
            ],
            "params": [
                "layer3.1.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 1024,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_97",
            "type": "BatchNorm",
            "inputs": [
                "537"
            ],
            "outputs": [
                "538"
            ],
            "params": [
                "layer3.1.bn3.running_mean",
                "layer3.1.bn3.running_var",
                "layer3.1.bn3.weight",
                "layer3.1.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_98",
            "type": "Elementwise",
            "inputs": [
                "538",
                "522"
            ],
            "outputs": [
                "544"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_100",
            "type": "Convolution2D",
            "inputs": [
                "544"
            ],
            "outputs": [
                "545"
            ],
            "params": [
                "layer3.2.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 256,
                "channels_in": 1024
            }
        },
        {
            "name": "BatchNormalization_101",
            "type": "BatchNorm",
            "inputs": [
                "545"
            ],
            "outputs": [
                "551"
            ],
            "params": [
                "layer3.2.bn1.running_mean",
                "layer3.2.bn1.running_var",
                "layer3.2.bn1.weight",
                "layer3.2.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_102",
            "type": "Activation",
            "inputs": [
                "551"
            ],
            "outputs": [
                "551"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_103",
            "type": "Convolution2D",
            "inputs": [
                "551"
            ],
            "outputs": [
                "552"
            ],
            "params": [
                "layer3.2.conv2.weight"
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
            "name": "BatchNormalization_104",
            "type": "BatchNorm",
            "inputs": [
                "552"
            ],
            "outputs": [
                "558"
            ],
            "params": [
                "layer3.2.bn2.running_mean",
                "layer3.2.bn2.running_var",
                "layer3.2.bn2.weight",
                "layer3.2.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_105",
            "type": "Activation",
            "inputs": [
                "558"
            ],
            "outputs": [
                "558"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_106",
            "type": "Convolution2D",
            "inputs": [
                "558"
            ],
            "outputs": [
                "559"
            ],
            "params": [
                "layer3.2.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 1024,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_107",
            "type": "BatchNorm",
            "inputs": [
                "559"
            ],
            "outputs": [
                "560"
            ],
            "params": [
                "layer3.2.bn3.running_mean",
                "layer3.2.bn3.running_var",
                "layer3.2.bn3.weight",
                "layer3.2.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_108",
            "type": "Elementwise",
            "inputs": [
                "560",
                "544"
            ],
            "outputs": [
                "566"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_110",
            "type": "Convolution2D",
            "inputs": [
                "566"
            ],
            "outputs": [
                "567"
            ],
            "params": [
                "layer3.3.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 256,
                "channels_in": 1024
            }
        },
        {
            "name": "BatchNormalization_111",
            "type": "BatchNorm",
            "inputs": [
                "567"
            ],
            "outputs": [
                "573"
            ],
            "params": [
                "layer3.3.bn1.running_mean",
                "layer3.3.bn1.running_var",
                "layer3.3.bn1.weight",
                "layer3.3.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_112",
            "type": "Activation",
            "inputs": [
                "573"
            ],
            "outputs": [
                "573"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_113",
            "type": "Convolution2D",
            "inputs": [
                "573"
            ],
            "outputs": [
                "574"
            ],
            "params": [
                "layer3.3.conv2.weight"
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
            "name": "BatchNormalization_114",
            "type": "BatchNorm",
            "inputs": [
                "574"
            ],
            "outputs": [
                "580"
            ],
            "params": [
                "layer3.3.bn2.running_mean",
                "layer3.3.bn2.running_var",
                "layer3.3.bn2.weight",
                "layer3.3.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_115",
            "type": "Activation",
            "inputs": [
                "580"
            ],
            "outputs": [
                "580"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_116",
            "type": "Convolution2D",
            "inputs": [
                "580"
            ],
            "outputs": [
                "581"
            ],
            "params": [
                "layer3.3.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 1024,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_117",
            "type": "BatchNorm",
            "inputs": [
                "581"
            ],
            "outputs": [
                "582"
            ],
            "params": [
                "layer3.3.bn3.running_mean",
                "layer3.3.bn3.running_var",
                "layer3.3.bn3.weight",
                "layer3.3.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_118",
            "type": "Elementwise",
            "inputs": [
                "582",
                "566"
            ],
            "outputs": [
                "588"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_120",
            "type": "Convolution2D",
            "inputs": [
                "588"
            ],
            "outputs": [
                "589"
            ],
            "params": [
                "layer3.4.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 256,
                "channels_in": 1024
            }
        },
        {
            "name": "BatchNormalization_121",
            "type": "BatchNorm",
            "inputs": [
                "589"
            ],
            "outputs": [
                "595"
            ],
            "params": [
                "layer3.4.bn1.running_mean",
                "layer3.4.bn1.running_var",
                "layer3.4.bn1.weight",
                "layer3.4.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_122",
            "type": "Activation",
            "inputs": [
                "595"
            ],
            "outputs": [
                "595"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_123",
            "type": "Convolution2D",
            "inputs": [
                "595"
            ],
            "outputs": [
                "596"
            ],
            "params": [
                "layer3.4.conv2.weight"
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
            "name": "BatchNormalization_124",
            "type": "BatchNorm",
            "inputs": [
                "596"
            ],
            "outputs": [
                "602"
            ],
            "params": [
                "layer3.4.bn2.running_mean",
                "layer3.4.bn2.running_var",
                "layer3.4.bn2.weight",
                "layer3.4.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_125",
            "type": "Activation",
            "inputs": [
                "602"
            ],
            "outputs": [
                "602"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_126",
            "type": "Convolution2D",
            "inputs": [
                "602"
            ],
            "outputs": [
                "603"
            ],
            "params": [
                "layer3.4.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 1024,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_127",
            "type": "BatchNorm",
            "inputs": [
                "603"
            ],
            "outputs": [
                "604"
            ],
            "params": [
                "layer3.4.bn3.running_mean",
                "layer3.4.bn3.running_var",
                "layer3.4.bn3.weight",
                "layer3.4.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_128",
            "type": "Elementwise",
            "inputs": [
                "604",
                "588"
            ],
            "outputs": [
                "610"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_130",
            "type": "Convolution2D",
            "inputs": [
                "610"
            ],
            "outputs": [
                "611"
            ],
            "params": [
                "layer3.5.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 256,
                "channels_in": 1024
            }
        },
        {
            "name": "BatchNormalization_131",
            "type": "BatchNorm",
            "inputs": [
                "611"
            ],
            "outputs": [
                "617"
            ],
            "params": [
                "layer3.5.bn1.running_mean",
                "layer3.5.bn1.running_var",
                "layer3.5.bn1.weight",
                "layer3.5.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_132",
            "type": "Activation",
            "inputs": [
                "617"
            ],
            "outputs": [
                "617"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_133",
            "type": "Convolution2D",
            "inputs": [
                "617"
            ],
            "outputs": [
                "618"
            ],
            "params": [
                "layer3.5.conv2.weight"
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
            "name": "BatchNormalization_134",
            "type": "BatchNorm",
            "inputs": [
                "618"
            ],
            "outputs": [
                "624"
            ],
            "params": [
                "layer3.5.bn2.running_mean",
                "layer3.5.bn2.running_var",
                "layer3.5.bn2.weight",
                "layer3.5.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_135",
            "type": "Activation",
            "inputs": [
                "624"
            ],
            "outputs": [
                "624"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_136",
            "type": "Convolution2D",
            "inputs": [
                "624"
            ],
            "outputs": [
                "625"
            ],
            "params": [
                "layer3.5.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 1024,
                "channels_in": 256
            }
        },
        {
            "name": "BatchNormalization_137",
            "type": "BatchNorm",
            "inputs": [
                "625"
            ],
            "outputs": [
                "626"
            ],
            "params": [
                "layer3.5.bn3.running_mean",
                "layer3.5.bn3.running_var",
                "layer3.5.bn3.weight",
                "layer3.5.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_138",
            "type": "Elementwise",
            "inputs": [
                "626",
                "610"
            ],
            "outputs": [
                "632"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_140",
            "type": "Convolution2D",
            "inputs": [
                "632"
            ],
            "outputs": [
                "633"
            ],
            "params": [
                "layer4.0.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 512,
                "channels_in": 1024
            }
        },
        {
            "name": "BatchNormalization_141",
            "type": "BatchNorm",
            "inputs": [
                "633"
            ],
            "outputs": [
                "639"
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
            "name": "Relu_142",
            "type": "Activation",
            "inputs": [
                "639"
            ],
            "outputs": [
                "639"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_143",
            "type": "Convolution2D",
            "inputs": [
                "639"
            ],
            "outputs": [
                "640"
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
                "channels_in": 512
            }
        },
        {
            "name": "BatchNormalization_144",
            "type": "BatchNorm",
            "inputs": [
                "640"
            ],
            "outputs": [
                "646"
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
            "name": "Relu_145",
            "type": "Activation",
            "inputs": [
                "646"
            ],
            "outputs": [
                "646"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_146",
            "type": "Convolution2D",
            "inputs": [
                "646"
            ],
            "outputs": [
                "647"
            ],
            "params": [
                "layer4.0.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 2048,
                "channels_in": 512
            }
        },
        {
            "name": "BatchNormalization_147",
            "type": "BatchNorm",
            "inputs": [
                "647"
            ],
            "outputs": [
                "648"
            ],
            "params": [
                "layer4.0.bn3.running_mean",
                "layer4.0.bn3.running_var",
                "layer4.0.bn3.weight",
                "layer4.0.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Conv_148",
            "type": "Convolution2D",
            "inputs": [
                "632"
            ],
            "outputs": [
                "653"
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
                "channels_out": 2048,
                "channels_in": 1024
            }
        },
        {
            "name": "BatchNormalization_149",
            "type": "BatchNorm",
            "inputs": [
                "653"
            ],
            "outputs": [
                "654"
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
            "name": "Add_150",
            "type": "Elementwise",
            "inputs": [
                "648",
                "654"
            ],
            "outputs": [
                "660"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_152",
            "type": "Convolution2D",
            "inputs": [
                "660"
            ],
            "outputs": [
                "661"
            ],
            "params": [
                "layer4.1.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 512,
                "channels_in": 2048
            }
        },
        {
            "name": "BatchNormalization_153",
            "type": "BatchNorm",
            "inputs": [
                "661"
            ],
            "outputs": [
                "667"
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
            "name": "Relu_154",
            "type": "Activation",
            "inputs": [
                "667"
            ],
            "outputs": [
                "667"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_155",
            "type": "Convolution2D",
            "inputs": [
                "667"
            ],
            "outputs": [
                "668"
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
            "name": "BatchNormalization_156",
            "type": "BatchNorm",
            "inputs": [
                "668"
            ],
            "outputs": [
                "674"
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
            "name": "Relu_157",
            "type": "Activation",
            "inputs": [
                "674"
            ],
            "outputs": [
                "674"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_158",
            "type": "Convolution2D",
            "inputs": [
                "674"
            ],
            "outputs": [
                "675"
            ],
            "params": [
                "layer4.1.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 2048,
                "channels_in": 512
            }
        },
        {
            "name": "BatchNormalization_159",
            "type": "BatchNorm",
            "inputs": [
                "675"
            ],
            "outputs": [
                "676"
            ],
            "params": [
                "layer4.1.bn3.running_mean",
                "layer4.1.bn3.running_var",
                "layer4.1.bn3.weight",
                "layer4.1.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_160",
            "type": "Elementwise",
            "inputs": [
                "676",
                "660"
            ],
            "outputs": [
                "682"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "Conv_162",
            "type": "Convolution2D",
            "inputs": [
                "682"
            ],
            "outputs": [
                "683"
            ],
            "params": [
                "layer4.2.conv1.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 512,
                "channels_in": 2048
            }
        },
        {
            "name": "BatchNormalization_163",
            "type": "BatchNorm",
            "inputs": [
                "683"
            ],
            "outputs": [
                "689"
            ],
            "params": [
                "layer4.2.bn1.running_mean",
                "layer4.2.bn1.running_var",
                "layer4.2.bn1.weight",
                "layer4.2.bn1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_164",
            "type": "Activation",
            "inputs": [
                "689"
            ],
            "outputs": [
                "689"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_165",
            "type": "Convolution2D",
            "inputs": [
                "689"
            ],
            "outputs": [
                "690"
            ],
            "params": [
                "layer4.2.conv2.weight"
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
            "name": "BatchNormalization_166",
            "type": "BatchNorm",
            "inputs": [
                "690"
            ],
            "outputs": [
                "696"
            ],
            "params": [
                "layer4.2.bn2.running_mean",
                "layer4.2.bn2.running_var",
                "layer4.2.bn2.weight",
                "layer4.2.bn2.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Relu_167",
            "type": "Activation",
            "inputs": [
                "696"
            ],
            "outputs": [
                "696"
            ],
            "options": {
                "activation": "relu"
            }
        },
        {
            "name": "Conv_168",
            "type": "Convolution2D",
            "inputs": [
                "696"
            ],
            "outputs": [
                "697"
            ],
            "params": [
                "layer4.2.conv3.weight"
            ],
            "options": {
                "bias": false,
                "groups": 1,
                "kernel": [
                    1,
                    1
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
                    0,
                    0
                ],
                "channels_out": 2048,
                "channels_in": 512
            }
        },
        {
            "name": "BatchNormalization_169",
            "type": "BatchNorm",
            "inputs": [
                "697"
            ],
            "outputs": [
                "698"
            ],
            "params": [
                "layer4.2.bn3.running_mean",
                "layer4.2.bn3.running_var",
                "layer4.2.bn3.weight",
                "layer4.2.bn3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_170",
            "type": "Elementwise",
            "inputs": [
                "698",
                "682"
            ],
            "outputs": [
                "704"
            ],
            "options": {
                "operation": "sum",
                "activation": "relu"
            }
        },
        {
            "name": "GlobalAveragePool_172",
            "type": "GlobalPooling",
            "inputs": [
                "704"
            ],
            "outputs": [
                "706"
            ],
            "options": {
                "mode": "avg"
            }
        },
        {
            "name": "Gemm_174",
            "type": "InnerProduct",
            "inputs": [
                "706"
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
                "inputs": 2048
            }
        }
    ]
}