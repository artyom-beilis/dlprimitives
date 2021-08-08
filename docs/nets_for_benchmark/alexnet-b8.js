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
                "18"
            ],
            "params": [
                "features.0.weight",
                "features.0.bias"
            ],
            "options": {
                "bias": true,
                "groups": 1,
                "kernel": [
                    11,
                    11
                ],
                "stride": [
                    4,
                    4
                ],
                "dilate": [
                    1,
                    1
                ],
                "pad": [
                    2,
                    2
                ],
                "channels_out": 64,
                "channels_in": 3,
                "activation": "relu"
            }
        },
        {
            "name": "MaxPool_2",
            "type": "Pooling2D",
            "inputs": [
                "18"
            ],
            "outputs": [
                "19"
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
                    0,
                    0
                ],
                "count_include_pad": 0,
                "mode": "max"
            }
        },
        {
            "name": "Conv_3",
            "type": "Convolution2D",
            "inputs": [
                "19"
            ],
            "outputs": [
                "21"
            ],
            "params": [
                "features.3.weight",
                "features.3.bias"
            ],
            "options": {
                "bias": true,
                "groups": 1,
                "kernel": [
                    5,
                    5
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
                    2,
                    2
                ],
                "channels_out": 192,
                "channels_in": 64,
                "activation": "relu"
            }
        },
        {
            "name": "MaxPool_5",
            "type": "Pooling2D",
            "inputs": [
                "21"
            ],
            "outputs": [
                "22"
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
                    0,
                    0
                ],
                "count_include_pad": 0,
                "mode": "max"
            }
        },
        {
            "name": "Conv_6",
            "type": "Convolution2D",
            "inputs": [
                "22"
            ],
            "outputs": [
                "24"
            ],
            "params": [
                "features.6.weight",
                "features.6.bias"
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
                "channels_out": 384,
                "channels_in": 192,
                "activation": "relu"
            }
        },
        {
            "name": "Conv_8",
            "type": "Convolution2D",
            "inputs": [
                "24"
            ],
            "outputs": [
                "26"
            ],
            "params": [
                "features.8.weight",
                "features.8.bias"
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
                "channels_in": 384,
                "activation": "relu"
            }
        },
        {
            "name": "Conv_10",
            "type": "Convolution2D",
            "inputs": [
                "26"
            ],
            "outputs": [
                "28"
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
                "channels_in": 256,
                "activation": "relu"
            }
        },
        {
            "name": "MaxPool_12",
            "type": "Pooling2D",
            "inputs": [
                "28"
            ],
            "outputs": [
                "29"
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
                    0,
                    0
                ],
                "count_include_pad": 0,
                "mode": "max"
            }
        },
        {
            "name": "AveragePool_13",
            "type": "Pooling2D",
            "inputs": [
                "29"
            ],
            "outputs": [
                "34"
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
            "name": "Gemm_18",
            "type": "InnerProduct",
            "inputs": [
                "34"
            ],
            "outputs": [
                "40"
            ],
            "params": [
                "classifier.1.weight",
                "classifier.1.bias"
            ],
            "options": {
                "bias": true,
                "outputs": 4096,
                "inputs": 9216,
                "activation": "relu"
            }
        },
        {
            "name": "Gemm_23",
            "type": "InnerProduct",
            "inputs": [
                "40"
            ],
            "outputs": [
                "43"
            ],
            "params": [
                "classifier.4.weight",
                "classifier.4.bias"
            ],
            "options": {
                "bias": true,
                "outputs": 4096,
                "inputs": 4096,
                "activation": "relu"
            }
        },
        {
            "name": "Gemm_25",
            "type": "InnerProduct",
            "inputs": [
                "43"
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