{
    "inputs": [
        {
            "shape": [
                8,
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
                "34"
            ],
            "params": [
                "features.0.weight",
                "features.0.bias"
            ],
            "options": {
                "bias": true,
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
                "channels_in": 3,
                "activation": "relu"
            }
        },
        {
            "name": "Conv_2",
            "type": "Convolution2D",
            "inputs": [
                "34"
            ],
            "outputs": [
                "36"
            ],
            "params": [
                "features.2.weight",
                "features.2.bias"
            ],
            "options": {
                "bias": true,
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
                "channels_in": 64,
                "activation": "relu"
            }
        },
        {
            "name": "MaxPool_4",
            "type": "Pooling2D",
            "inputs": [
                "36"
            ],
            "outputs": [
                "37"
            ],
            "options": {
                "kernel": [
                    2,
                    2
                ],
                "stride": [
                    2,
                    2
                ],
                "pad": [
                    0,
                    0
                ],
                "count_include_pad": 0,
                "mode": "max"
            }
        },
        {
            "name": "Conv_5",
            "type": "Convolution2D",
            "inputs": [
                "37"
            ],
            "outputs": [
                "39"
            ],
            "params": [
                "features.5.weight",
                "features.5.bias"
            ],
            "options": {
                "bias": true,
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
                "channels_in": 64,
                "activation": "relu"
            }
        },
        {
            "name": "Conv_7",
            "type": "Convolution2D",
            "inputs": [
                "39"
            ],
            "outputs": [
                "41"
            ],
            "params": [
                "features.7.weight",
                "features.7.bias"
            ],
            "options": {
                "bias": true,
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
                "channels_in": 128,
                "activation": "relu"
            }
        },
        {
            "name": "MaxPool_9",
            "type": "Pooling2D",
            "inputs": [
                "41"
            ],
            "outputs": [
                "42"
            ],
            "options": {
                "kernel": [
                    2,
                    2
                ],
                "stride": [
                    2,
                    2
                ],
                "pad": [
                    0,
                    0
                ],
                "count_include_pad": 0,
                "mode": "max"
            }
        },
        {
            "name": "Conv_10",
            "type": "Convolution2D",
            "inputs": [
                "42"
            ],
            "outputs": [
                "44"
            ],
            "params": [
                "features.10.weight",
                "features.10.bias"
            ],
            "options": {
                "bias": true,
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
                "channels_in": 128,
                "activation": "relu"
            }
        },
        {
            "name": "Conv_12",
            "type": "Convolution2D",
            "inputs": [
                "44"
            ],
            "outputs": [
                "46"
            ],
            "params": [
                "features.12.weight",
                "features.12.bias"
            ],
            "options": {
                "bias": true,
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
                "channels_in": 256,
                "activation": "relu"
            }
        },
        {
            "name": "Conv_14",
            "type": "Convolution2D",
            "inputs": [
                "46"
            ],
            "outputs": [
                "48"
            ],
            "params": [
                "features.14.weight",
                "features.14.bias"
            ],
            "options": {
                "bias": true,
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
                "channels_in": 256,
                "activation": "relu"
            }
        },
        {
            "name": "MaxPool_16",
            "type": "Pooling2D",
            "inputs": [
                "48"
            ],
            "outputs": [
                "49"
            ],
            "options": {
                "kernel": [
                    2,
                    2
                ],
                "stride": [
                    2,
                    2
                ],
                "pad": [
                    0,
                    0
                ],
                "count_include_pad": 0,
                "mode": "max"
            }
        },
        {
            "name": "Conv_17",
            "type": "Convolution2D",
            "inputs": [
                "49"
            ],
            "outputs": [
                "51"
            ],
            "params": [
                "features.17.weight",
                "features.17.bias"
            ],
            "options": {
                "bias": true,
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
                "channels_in": 256,
                "activation": "relu"
            }
        },
        {
            "name": "Conv_19",
            "type": "Convolution2D",
            "inputs": [
                "51"
            ],
            "outputs": [
                "53"
            ],
            "params": [
                "features.19.weight",
                "features.19.bias"
            ],
            "options": {
                "bias": true,
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
                "channels_in": 512,
                "activation": "relu"
            }
        },
        {
            "name": "Conv_21",
            "type": "Convolution2D",
            "inputs": [
                "53"
            ],
            "outputs": [
                "55"
            ],
            "params": [
                "features.21.weight",
                "features.21.bias"
            ],
            "options": {
                "bias": true,
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
                "channels_in": 512,
                "activation": "relu"
            }
        },
        {
            "name": "MaxPool_23",
            "type": "Pooling2D",
            "inputs": [
                "55"
            ],
            "outputs": [
                "56"
            ],
            "options": {
                "kernel": [
                    2,
                    2
                ],
                "stride": [
                    2,
                    2
                ],
                "pad": [
                    0,
                    0
                ],
                "count_include_pad": 0,
                "mode": "max"
            }
        },
        {
            "name": "Conv_24",
            "type": "Convolution2D",
            "inputs": [
                "56"
            ],
            "outputs": [
                "58"
            ],
            "params": [
                "features.24.weight",
                "features.24.bias"
            ],
            "options": {
                "bias": true,
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
                "channels_in": 512,
                "activation": "relu"
            }
        },
        {
            "name": "Conv_26",
            "type": "Convolution2D",
            "inputs": [
                "58"
            ],
            "outputs": [
                "60"
            ],
            "params": [
                "features.26.weight",
                "features.26.bias"
            ],
            "options": {
                "bias": true,
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
                "channels_in": 512,
                "activation": "relu"
            }
        },
        {
            "name": "Conv_28",
            "type": "Convolution2D",
            "inputs": [
                "60"
            ],
            "outputs": [
                "62"
            ],
            "params": [
                "features.28.weight",
                "features.28.bias"
            ],
            "options": {
                "bias": true,
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
                "channels_in": 512,
                "activation": "relu"
            }
        },
        {
            "name": "MaxPool_30",
            "type": "Pooling2D",
            "inputs": [
                "62"
            ],
            "outputs": [
                "63"
            ],
            "options": {
                "kernel": [
                    2,
                    2
                ],
                "stride": [
                    2,
                    2
                ],
                "pad": [
                    0,
                    0
                ],
                "count_include_pad": 0,
                "mode": "max"
            }
        },
        {
            "name": "AveragePool_31",
            "type": "Pooling2D",
            "inputs": [
                "63"
            ],
            "outputs": [
                "65"
            ],
            "options": {
                "kernel": [
                    1,
                    1
                ],
                "stride": [
                    1,
                    1
                ],
                "pad": [
                    0,
                    0
                ],
                "count_include_pad": 0,
                "mode": "avg"
            }
        },
        {
            "name": "Gemm_33",
            "type": "InnerProduct",
            "inputs": [
                "65"
            ],
            "outputs": [
                "70"
            ],
            "params": [
                "classifier.0.weight",
                "classifier.0.bias"
            ],
            "options": {
                "bias": true,
                "outputs": 4096,
                "inputs": 25088,
                "activation": "relu"
            }
        },
        {
            "name": "Gemm_38",
            "type": "InnerProduct",
            "inputs": [
                "70"
            ],
            "outputs": [
                "76"
            ],
            "params": [
                "classifier.3.weight",
                "classifier.3.bias"
            ],
            "options": {
                "bias": true,
                "outputs": 4096,
                "inputs": 4096,
                "activation": "relu"
            }
        },
        {
            "name": "Gemm_43",
            "type": "InnerProduct",
            "inputs": [
                "76"
            ],
            "outputs": [
                "loss"
            ],
            "params": [
                "classifier.6.weight",
                "classifier.6.bias"
            ],
            "options": {
                "bias": true,
                "outputs": 1000,
                "inputs": 4096
            }
        }
    ]
}