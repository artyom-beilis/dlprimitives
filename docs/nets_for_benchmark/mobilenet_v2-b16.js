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
                "315"
            ],
            "params": [
                "features.0.0.weight"
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
                "channels_out": 32,
                "channels_in": 3
            }
        },
        {
            "name": "BatchNormalization_1",
            "type": "BatchNorm",
            "inputs": [
                "315"
            ],
            "outputs": [
                "323"
            ],
            "params": [
                "features.0.1.running_mean",
                "features.0.1.running_var",
                "features.0.1.weight",
                "features.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_4",
            "type": "Activation",
            "inputs": [
                "323"
            ],
            "outputs": [
                "323"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_5",
            "type": "Convolution2D",
            "inputs": [
                "323"
            ],
            "outputs": [
                "324"
            ],
            "params": [
                "features.1.conv.0.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 32,
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
                "channels_out": 32,
                "channels_in": 32
            }
        },
        {
            "name": "BatchNormalization_6",
            "type": "BatchNorm",
            "inputs": [
                "324"
            ],
            "outputs": [
                "332"
            ],
            "params": [
                "features.1.conv.0.1.running_mean",
                "features.1.conv.0.1.running_var",
                "features.1.conv.0.1.weight",
                "features.1.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_9",
            "type": "Activation",
            "inputs": [
                "332"
            ],
            "outputs": [
                "332"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_10",
            "type": "Convolution2D",
            "inputs": [
                "332"
            ],
            "outputs": [
                "333"
            ],
            "params": [
                "features.1.conv.1.weight"
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
                "channels_out": 16,
                "channels_in": 32
            }
        },
        {
            "name": "BatchNormalization_11",
            "type": "BatchNorm",
            "inputs": [
                "333"
            ],
            "outputs": [
                "334"
            ],
            "params": [
                "features.1.conv.2.running_mean",
                "features.1.conv.2.running_var",
                "features.1.conv.2.weight",
                "features.1.conv.2.bias"
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
                "334"
            ],
            "outputs": [
                "339"
            ],
            "params": [
                "features.2.conv.0.0.weight"
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
                "channels_out": 96,
                "channels_in": 16
            }
        },
        {
            "name": "BatchNormalization_13",
            "type": "BatchNorm",
            "inputs": [
                "339"
            ],
            "outputs": [
                "347"
            ],
            "params": [
                "features.2.conv.0.1.running_mean",
                "features.2.conv.0.1.running_var",
                "features.2.conv.0.1.weight",
                "features.2.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_16",
            "type": "Activation",
            "inputs": [
                "347"
            ],
            "outputs": [
                "347"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_17",
            "type": "Convolution2D",
            "inputs": [
                "347"
            ],
            "outputs": [
                "348"
            ],
            "params": [
                "features.2.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 96,
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
                "channels_out": 96,
                "channels_in": 96
            }
        },
        {
            "name": "BatchNormalization_18",
            "type": "BatchNorm",
            "inputs": [
                "348"
            ],
            "outputs": [
                "356"
            ],
            "params": [
                "features.2.conv.1.1.running_mean",
                "features.2.conv.1.1.running_var",
                "features.2.conv.1.1.weight",
                "features.2.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_21",
            "type": "Activation",
            "inputs": [
                "356"
            ],
            "outputs": [
                "356"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_22",
            "type": "Convolution2D",
            "inputs": [
                "356"
            ],
            "outputs": [
                "357"
            ],
            "params": [
                "features.2.conv.2.weight"
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
                "channels_out": 24,
                "channels_in": 96
            }
        },
        {
            "name": "BatchNormalization_23",
            "type": "BatchNorm",
            "inputs": [
                "357"
            ],
            "outputs": [
                "358"
            ],
            "params": [
                "features.2.conv.3.running_mean",
                "features.2.conv.3.running_var",
                "features.2.conv.3.weight",
                "features.2.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Conv_24",
            "type": "Convolution2D",
            "inputs": [
                "358"
            ],
            "outputs": [
                "363"
            ],
            "params": [
                "features.3.conv.0.0.weight"
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
                "channels_out": 144,
                "channels_in": 24
            }
        },
        {
            "name": "BatchNormalization_25",
            "type": "BatchNorm",
            "inputs": [
                "363"
            ],
            "outputs": [
                "371"
            ],
            "params": [
                "features.3.conv.0.1.running_mean",
                "features.3.conv.0.1.running_var",
                "features.3.conv.0.1.weight",
                "features.3.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_28",
            "type": "Activation",
            "inputs": [
                "371"
            ],
            "outputs": [
                "371"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_29",
            "type": "Convolution2D",
            "inputs": [
                "371"
            ],
            "outputs": [
                "372"
            ],
            "params": [
                "features.3.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 144,
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
                "channels_out": 144,
                "channels_in": 144
            }
        },
        {
            "name": "BatchNormalization_30",
            "type": "BatchNorm",
            "inputs": [
                "372"
            ],
            "outputs": [
                "380"
            ],
            "params": [
                "features.3.conv.1.1.running_mean",
                "features.3.conv.1.1.running_var",
                "features.3.conv.1.1.weight",
                "features.3.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_33",
            "type": "Activation",
            "inputs": [
                "380"
            ],
            "outputs": [
                "380"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_34",
            "type": "Convolution2D",
            "inputs": [
                "380"
            ],
            "outputs": [
                "381"
            ],
            "params": [
                "features.3.conv.2.weight"
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
                "channels_out": 24,
                "channels_in": 144
            }
        },
        {
            "name": "BatchNormalization_35",
            "type": "BatchNorm",
            "inputs": [
                "381"
            ],
            "outputs": [
                "382"
            ],
            "params": [
                "features.3.conv.3.running_mean",
                "features.3.conv.3.running_var",
                "features.3.conv.3.weight",
                "features.3.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_36",
            "type": "Elementwise",
            "inputs": [
                "358",
                "382"
            ],
            "outputs": [
                "387"
            ],
            "options": {
                "operation": "sum"
            }
        },
        {
            "name": "Conv_37",
            "type": "Convolution2D",
            "inputs": [
                "387"
            ],
            "outputs": [
                "388"
            ],
            "params": [
                "features.4.conv.0.0.weight"
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
                "channels_out": 144,
                "channels_in": 24
            }
        },
        {
            "name": "BatchNormalization_38",
            "type": "BatchNorm",
            "inputs": [
                "388"
            ],
            "outputs": [
                "396"
            ],
            "params": [
                "features.4.conv.0.1.running_mean",
                "features.4.conv.0.1.running_var",
                "features.4.conv.0.1.weight",
                "features.4.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_41",
            "type": "Activation",
            "inputs": [
                "396"
            ],
            "outputs": [
                "396"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_42",
            "type": "Convolution2D",
            "inputs": [
                "396"
            ],
            "outputs": [
                "397"
            ],
            "params": [
                "features.4.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 144,
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
                "channels_out": 144,
                "channels_in": 144
            }
        },
        {
            "name": "BatchNormalization_43",
            "type": "BatchNorm",
            "inputs": [
                "397"
            ],
            "outputs": [
                "405"
            ],
            "params": [
                "features.4.conv.1.1.running_mean",
                "features.4.conv.1.1.running_var",
                "features.4.conv.1.1.weight",
                "features.4.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_46",
            "type": "Activation",
            "inputs": [
                "405"
            ],
            "outputs": [
                "405"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_47",
            "type": "Convolution2D",
            "inputs": [
                "405"
            ],
            "outputs": [
                "406"
            ],
            "params": [
                "features.4.conv.2.weight"
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
                "channels_out": 32,
                "channels_in": 144
            }
        },
        {
            "name": "BatchNormalization_48",
            "type": "BatchNorm",
            "inputs": [
                "406"
            ],
            "outputs": [
                "407"
            ],
            "params": [
                "features.4.conv.3.running_mean",
                "features.4.conv.3.running_var",
                "features.4.conv.3.weight",
                "features.4.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Conv_49",
            "type": "Convolution2D",
            "inputs": [
                "407"
            ],
            "outputs": [
                "412"
            ],
            "params": [
                "features.5.conv.0.0.weight"
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
                "channels_out": 192,
                "channels_in": 32
            }
        },
        {
            "name": "BatchNormalization_50",
            "type": "BatchNorm",
            "inputs": [
                "412"
            ],
            "outputs": [
                "420"
            ],
            "params": [
                "features.5.conv.0.1.running_mean",
                "features.5.conv.0.1.running_var",
                "features.5.conv.0.1.weight",
                "features.5.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_53",
            "type": "Activation",
            "inputs": [
                "420"
            ],
            "outputs": [
                "420"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_54",
            "type": "Convolution2D",
            "inputs": [
                "420"
            ],
            "outputs": [
                "421"
            ],
            "params": [
                "features.5.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 192,
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
                "channels_out": 192,
                "channels_in": 192
            }
        },
        {
            "name": "BatchNormalization_55",
            "type": "BatchNorm",
            "inputs": [
                "421"
            ],
            "outputs": [
                "429"
            ],
            "params": [
                "features.5.conv.1.1.running_mean",
                "features.5.conv.1.1.running_var",
                "features.5.conv.1.1.weight",
                "features.5.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_58",
            "type": "Activation",
            "inputs": [
                "429"
            ],
            "outputs": [
                "429"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_59",
            "type": "Convolution2D",
            "inputs": [
                "429"
            ],
            "outputs": [
                "430"
            ],
            "params": [
                "features.5.conv.2.weight"
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
                "channels_out": 32,
                "channels_in": 192
            }
        },
        {
            "name": "BatchNormalization_60",
            "type": "BatchNorm",
            "inputs": [
                "430"
            ],
            "outputs": [
                "431"
            ],
            "params": [
                "features.5.conv.3.running_mean",
                "features.5.conv.3.running_var",
                "features.5.conv.3.weight",
                "features.5.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_61",
            "type": "Elementwise",
            "inputs": [
                "407",
                "431"
            ],
            "outputs": [
                "436"
            ],
            "options": {
                "operation": "sum"
            }
        },
        {
            "name": "Conv_62",
            "type": "Convolution2D",
            "inputs": [
                "436"
            ],
            "outputs": [
                "437"
            ],
            "params": [
                "features.6.conv.0.0.weight"
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
                "channels_out": 192,
                "channels_in": 32
            }
        },
        {
            "name": "BatchNormalization_63",
            "type": "BatchNorm",
            "inputs": [
                "437"
            ],
            "outputs": [
                "445"
            ],
            "params": [
                "features.6.conv.0.1.running_mean",
                "features.6.conv.0.1.running_var",
                "features.6.conv.0.1.weight",
                "features.6.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_66",
            "type": "Activation",
            "inputs": [
                "445"
            ],
            "outputs": [
                "445"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_67",
            "type": "Convolution2D",
            "inputs": [
                "445"
            ],
            "outputs": [
                "446"
            ],
            "params": [
                "features.6.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 192,
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
                "channels_out": 192,
                "channels_in": 192
            }
        },
        {
            "name": "BatchNormalization_68",
            "type": "BatchNorm",
            "inputs": [
                "446"
            ],
            "outputs": [
                "454"
            ],
            "params": [
                "features.6.conv.1.1.running_mean",
                "features.6.conv.1.1.running_var",
                "features.6.conv.1.1.weight",
                "features.6.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_71",
            "type": "Activation",
            "inputs": [
                "454"
            ],
            "outputs": [
                "454"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_72",
            "type": "Convolution2D",
            "inputs": [
                "454"
            ],
            "outputs": [
                "455"
            ],
            "params": [
                "features.6.conv.2.weight"
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
                "channels_out": 32,
                "channels_in": 192
            }
        },
        {
            "name": "BatchNormalization_73",
            "type": "BatchNorm",
            "inputs": [
                "455"
            ],
            "outputs": [
                "456"
            ],
            "params": [
                "features.6.conv.3.running_mean",
                "features.6.conv.3.running_var",
                "features.6.conv.3.weight",
                "features.6.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_74",
            "type": "Elementwise",
            "inputs": [
                "436",
                "456"
            ],
            "outputs": [
                "461"
            ],
            "options": {
                "operation": "sum"
            }
        },
        {
            "name": "Conv_75",
            "type": "Convolution2D",
            "inputs": [
                "461"
            ],
            "outputs": [
                "462"
            ],
            "params": [
                "features.7.conv.0.0.weight"
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
                "channels_out": 192,
                "channels_in": 32
            }
        },
        {
            "name": "BatchNormalization_76",
            "type": "BatchNorm",
            "inputs": [
                "462"
            ],
            "outputs": [
                "470"
            ],
            "params": [
                "features.7.conv.0.1.running_mean",
                "features.7.conv.0.1.running_var",
                "features.7.conv.0.1.weight",
                "features.7.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_79",
            "type": "Activation",
            "inputs": [
                "470"
            ],
            "outputs": [
                "470"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_80",
            "type": "Convolution2D",
            "inputs": [
                "470"
            ],
            "outputs": [
                "471"
            ],
            "params": [
                "features.7.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 192,
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
                "channels_out": 192,
                "channels_in": 192
            }
        },
        {
            "name": "BatchNormalization_81",
            "type": "BatchNorm",
            "inputs": [
                "471"
            ],
            "outputs": [
                "479"
            ],
            "params": [
                "features.7.conv.1.1.running_mean",
                "features.7.conv.1.1.running_var",
                "features.7.conv.1.1.weight",
                "features.7.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_84",
            "type": "Activation",
            "inputs": [
                "479"
            ],
            "outputs": [
                "479"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_85",
            "type": "Convolution2D",
            "inputs": [
                "479"
            ],
            "outputs": [
                "480"
            ],
            "params": [
                "features.7.conv.2.weight"
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
                "channels_in": 192
            }
        },
        {
            "name": "BatchNormalization_86",
            "type": "BatchNorm",
            "inputs": [
                "480"
            ],
            "outputs": [
                "481"
            ],
            "params": [
                "features.7.conv.3.running_mean",
                "features.7.conv.3.running_var",
                "features.7.conv.3.weight",
                "features.7.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Conv_87",
            "type": "Convolution2D",
            "inputs": [
                "481"
            ],
            "outputs": [
                "486"
            ],
            "params": [
                "features.8.conv.0.0.weight"
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
                "channels_out": 384,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_88",
            "type": "BatchNorm",
            "inputs": [
                "486"
            ],
            "outputs": [
                "494"
            ],
            "params": [
                "features.8.conv.0.1.running_mean",
                "features.8.conv.0.1.running_var",
                "features.8.conv.0.1.weight",
                "features.8.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_91",
            "type": "Activation",
            "inputs": [
                "494"
            ],
            "outputs": [
                "494"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_92",
            "type": "Convolution2D",
            "inputs": [
                "494"
            ],
            "outputs": [
                "495"
            ],
            "params": [
                "features.8.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 384,
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
                "channels_in": 384
            }
        },
        {
            "name": "BatchNormalization_93",
            "type": "BatchNorm",
            "inputs": [
                "495"
            ],
            "outputs": [
                "503"
            ],
            "params": [
                "features.8.conv.1.1.running_mean",
                "features.8.conv.1.1.running_var",
                "features.8.conv.1.1.weight",
                "features.8.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_96",
            "type": "Activation",
            "inputs": [
                "503"
            ],
            "outputs": [
                "503"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_97",
            "type": "Convolution2D",
            "inputs": [
                "503"
            ],
            "outputs": [
                "504"
            ],
            "params": [
                "features.8.conv.2.weight"
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
                "channels_in": 384
            }
        },
        {
            "name": "BatchNormalization_98",
            "type": "BatchNorm",
            "inputs": [
                "504"
            ],
            "outputs": [
                "505"
            ],
            "params": [
                "features.8.conv.3.running_mean",
                "features.8.conv.3.running_var",
                "features.8.conv.3.weight",
                "features.8.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_99",
            "type": "Elementwise",
            "inputs": [
                "481",
                "505"
            ],
            "outputs": [
                "510"
            ],
            "options": {
                "operation": "sum"
            }
        },
        {
            "name": "Conv_100",
            "type": "Convolution2D",
            "inputs": [
                "510"
            ],
            "outputs": [
                "511"
            ],
            "params": [
                "features.9.conv.0.0.weight"
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
                "channels_out": 384,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_101",
            "type": "BatchNorm",
            "inputs": [
                "511"
            ],
            "outputs": [
                "519"
            ],
            "params": [
                "features.9.conv.0.1.running_mean",
                "features.9.conv.0.1.running_var",
                "features.9.conv.0.1.weight",
                "features.9.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_104",
            "type": "Activation",
            "inputs": [
                "519"
            ],
            "outputs": [
                "519"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_105",
            "type": "Convolution2D",
            "inputs": [
                "519"
            ],
            "outputs": [
                "520"
            ],
            "params": [
                "features.9.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 384,
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
                "channels_in": 384
            }
        },
        {
            "name": "BatchNormalization_106",
            "type": "BatchNorm",
            "inputs": [
                "520"
            ],
            "outputs": [
                "528"
            ],
            "params": [
                "features.9.conv.1.1.running_mean",
                "features.9.conv.1.1.running_var",
                "features.9.conv.1.1.weight",
                "features.9.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_109",
            "type": "Activation",
            "inputs": [
                "528"
            ],
            "outputs": [
                "528"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_110",
            "type": "Convolution2D",
            "inputs": [
                "528"
            ],
            "outputs": [
                "529"
            ],
            "params": [
                "features.9.conv.2.weight"
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
                "channels_in": 384
            }
        },
        {
            "name": "BatchNormalization_111",
            "type": "BatchNorm",
            "inputs": [
                "529"
            ],
            "outputs": [
                "530"
            ],
            "params": [
                "features.9.conv.3.running_mean",
                "features.9.conv.3.running_var",
                "features.9.conv.3.weight",
                "features.9.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_112",
            "type": "Elementwise",
            "inputs": [
                "510",
                "530"
            ],
            "outputs": [
                "535"
            ],
            "options": {
                "operation": "sum"
            }
        },
        {
            "name": "Conv_113",
            "type": "Convolution2D",
            "inputs": [
                "535"
            ],
            "outputs": [
                "536"
            ],
            "params": [
                "features.10.conv.0.0.weight"
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
                "channels_out": 384,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_114",
            "type": "BatchNorm",
            "inputs": [
                "536"
            ],
            "outputs": [
                "544"
            ],
            "params": [
                "features.10.conv.0.1.running_mean",
                "features.10.conv.0.1.running_var",
                "features.10.conv.0.1.weight",
                "features.10.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_117",
            "type": "Activation",
            "inputs": [
                "544"
            ],
            "outputs": [
                "544"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_118",
            "type": "Convolution2D",
            "inputs": [
                "544"
            ],
            "outputs": [
                "545"
            ],
            "params": [
                "features.10.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 384,
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
                "channels_in": 384
            }
        },
        {
            "name": "BatchNormalization_119",
            "type": "BatchNorm",
            "inputs": [
                "545"
            ],
            "outputs": [
                "553"
            ],
            "params": [
                "features.10.conv.1.1.running_mean",
                "features.10.conv.1.1.running_var",
                "features.10.conv.1.1.weight",
                "features.10.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_122",
            "type": "Activation",
            "inputs": [
                "553"
            ],
            "outputs": [
                "553"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_123",
            "type": "Convolution2D",
            "inputs": [
                "553"
            ],
            "outputs": [
                "554"
            ],
            "params": [
                "features.10.conv.2.weight"
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
                "channels_in": 384
            }
        },
        {
            "name": "BatchNormalization_124",
            "type": "BatchNorm",
            "inputs": [
                "554"
            ],
            "outputs": [
                "555"
            ],
            "params": [
                "features.10.conv.3.running_mean",
                "features.10.conv.3.running_var",
                "features.10.conv.3.weight",
                "features.10.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_125",
            "type": "Elementwise",
            "inputs": [
                "535",
                "555"
            ],
            "outputs": [
                "560"
            ],
            "options": {
                "operation": "sum"
            }
        },
        {
            "name": "Conv_126",
            "type": "Convolution2D",
            "inputs": [
                "560"
            ],
            "outputs": [
                "561"
            ],
            "params": [
                "features.11.conv.0.0.weight"
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
                "channels_out": 384,
                "channels_in": 64
            }
        },
        {
            "name": "BatchNormalization_127",
            "type": "BatchNorm",
            "inputs": [
                "561"
            ],
            "outputs": [
                "569"
            ],
            "params": [
                "features.11.conv.0.1.running_mean",
                "features.11.conv.0.1.running_var",
                "features.11.conv.0.1.weight",
                "features.11.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_130",
            "type": "Activation",
            "inputs": [
                "569"
            ],
            "outputs": [
                "569"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_131",
            "type": "Convolution2D",
            "inputs": [
                "569"
            ],
            "outputs": [
                "570"
            ],
            "params": [
                "features.11.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 384,
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
                "channels_in": 384
            }
        },
        {
            "name": "BatchNormalization_132",
            "type": "BatchNorm",
            "inputs": [
                "570"
            ],
            "outputs": [
                "578"
            ],
            "params": [
                "features.11.conv.1.1.running_mean",
                "features.11.conv.1.1.running_var",
                "features.11.conv.1.1.weight",
                "features.11.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_135",
            "type": "Activation",
            "inputs": [
                "578"
            ],
            "outputs": [
                "578"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_136",
            "type": "Convolution2D",
            "inputs": [
                "578"
            ],
            "outputs": [
                "579"
            ],
            "params": [
                "features.11.conv.2.weight"
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
                "channels_out": 96,
                "channels_in": 384
            }
        },
        {
            "name": "BatchNormalization_137",
            "type": "BatchNorm",
            "inputs": [
                "579"
            ],
            "outputs": [
                "580"
            ],
            "params": [
                "features.11.conv.3.running_mean",
                "features.11.conv.3.running_var",
                "features.11.conv.3.weight",
                "features.11.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Conv_138",
            "type": "Convolution2D",
            "inputs": [
                "580"
            ],
            "outputs": [
                "585"
            ],
            "params": [
                "features.12.conv.0.0.weight"
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
                "channels_out": 576,
                "channels_in": 96
            }
        },
        {
            "name": "BatchNormalization_139",
            "type": "BatchNorm",
            "inputs": [
                "585"
            ],
            "outputs": [
                "593"
            ],
            "params": [
                "features.12.conv.0.1.running_mean",
                "features.12.conv.0.1.running_var",
                "features.12.conv.0.1.weight",
                "features.12.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_142",
            "type": "Activation",
            "inputs": [
                "593"
            ],
            "outputs": [
                "593"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_143",
            "type": "Convolution2D",
            "inputs": [
                "593"
            ],
            "outputs": [
                "594"
            ],
            "params": [
                "features.12.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 576,
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
                "channels_out": 576,
                "channels_in": 576
            }
        },
        {
            "name": "BatchNormalization_144",
            "type": "BatchNorm",
            "inputs": [
                "594"
            ],
            "outputs": [
                "602"
            ],
            "params": [
                "features.12.conv.1.1.running_mean",
                "features.12.conv.1.1.running_var",
                "features.12.conv.1.1.weight",
                "features.12.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_147",
            "type": "Activation",
            "inputs": [
                "602"
            ],
            "outputs": [
                "602"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_148",
            "type": "Convolution2D",
            "inputs": [
                "602"
            ],
            "outputs": [
                "603"
            ],
            "params": [
                "features.12.conv.2.weight"
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
                "channels_out": 96,
                "channels_in": 576
            }
        },
        {
            "name": "BatchNormalization_149",
            "type": "BatchNorm",
            "inputs": [
                "603"
            ],
            "outputs": [
                "604"
            ],
            "params": [
                "features.12.conv.3.running_mean",
                "features.12.conv.3.running_var",
                "features.12.conv.3.weight",
                "features.12.conv.3.bias"
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
                "580",
                "604"
            ],
            "outputs": [
                "609"
            ],
            "options": {
                "operation": "sum"
            }
        },
        {
            "name": "Conv_151",
            "type": "Convolution2D",
            "inputs": [
                "609"
            ],
            "outputs": [
                "610"
            ],
            "params": [
                "features.13.conv.0.0.weight"
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
                "channels_out": 576,
                "channels_in": 96
            }
        },
        {
            "name": "BatchNormalization_152",
            "type": "BatchNorm",
            "inputs": [
                "610"
            ],
            "outputs": [
                "618"
            ],
            "params": [
                "features.13.conv.0.1.running_mean",
                "features.13.conv.0.1.running_var",
                "features.13.conv.0.1.weight",
                "features.13.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_155",
            "type": "Activation",
            "inputs": [
                "618"
            ],
            "outputs": [
                "618"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_156",
            "type": "Convolution2D",
            "inputs": [
                "618"
            ],
            "outputs": [
                "619"
            ],
            "params": [
                "features.13.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 576,
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
                "channels_out": 576,
                "channels_in": 576
            }
        },
        {
            "name": "BatchNormalization_157",
            "type": "BatchNorm",
            "inputs": [
                "619"
            ],
            "outputs": [
                "627"
            ],
            "params": [
                "features.13.conv.1.1.running_mean",
                "features.13.conv.1.1.running_var",
                "features.13.conv.1.1.weight",
                "features.13.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_160",
            "type": "Activation",
            "inputs": [
                "627"
            ],
            "outputs": [
                "627"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_161",
            "type": "Convolution2D",
            "inputs": [
                "627"
            ],
            "outputs": [
                "628"
            ],
            "params": [
                "features.13.conv.2.weight"
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
                "channels_out": 96,
                "channels_in": 576
            }
        },
        {
            "name": "BatchNormalization_162",
            "type": "BatchNorm",
            "inputs": [
                "628"
            ],
            "outputs": [
                "629"
            ],
            "params": [
                "features.13.conv.3.running_mean",
                "features.13.conv.3.running_var",
                "features.13.conv.3.weight",
                "features.13.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_163",
            "type": "Elementwise",
            "inputs": [
                "609",
                "629"
            ],
            "outputs": [
                "634"
            ],
            "options": {
                "operation": "sum"
            }
        },
        {
            "name": "Conv_164",
            "type": "Convolution2D",
            "inputs": [
                "634"
            ],
            "outputs": [
                "635"
            ],
            "params": [
                "features.14.conv.0.0.weight"
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
                "channels_out": 576,
                "channels_in": 96
            }
        },
        {
            "name": "BatchNormalization_165",
            "type": "BatchNorm",
            "inputs": [
                "635"
            ],
            "outputs": [
                "643"
            ],
            "params": [
                "features.14.conv.0.1.running_mean",
                "features.14.conv.0.1.running_var",
                "features.14.conv.0.1.weight",
                "features.14.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_168",
            "type": "Activation",
            "inputs": [
                "643"
            ],
            "outputs": [
                "643"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_169",
            "type": "Convolution2D",
            "inputs": [
                "643"
            ],
            "outputs": [
                "644"
            ],
            "params": [
                "features.14.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 576,
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
                "channels_out": 576,
                "channels_in": 576
            }
        },
        {
            "name": "BatchNormalization_170",
            "type": "BatchNorm",
            "inputs": [
                "644"
            ],
            "outputs": [
                "652"
            ],
            "params": [
                "features.14.conv.1.1.running_mean",
                "features.14.conv.1.1.running_var",
                "features.14.conv.1.1.weight",
                "features.14.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_173",
            "type": "Activation",
            "inputs": [
                "652"
            ],
            "outputs": [
                "652"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_174",
            "type": "Convolution2D",
            "inputs": [
                "652"
            ],
            "outputs": [
                "653"
            ],
            "params": [
                "features.14.conv.2.weight"
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
                "channels_out": 160,
                "channels_in": 576
            }
        },
        {
            "name": "BatchNormalization_175",
            "type": "BatchNorm",
            "inputs": [
                "653"
            ],
            "outputs": [
                "654"
            ],
            "params": [
                "features.14.conv.3.running_mean",
                "features.14.conv.3.running_var",
                "features.14.conv.3.weight",
                "features.14.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Conv_176",
            "type": "Convolution2D",
            "inputs": [
                "654"
            ],
            "outputs": [
                "659"
            ],
            "params": [
                "features.15.conv.0.0.weight"
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
                "channels_out": 960,
                "channels_in": 160
            }
        },
        {
            "name": "BatchNormalization_177",
            "type": "BatchNorm",
            "inputs": [
                "659"
            ],
            "outputs": [
                "667"
            ],
            "params": [
                "features.15.conv.0.1.running_mean",
                "features.15.conv.0.1.running_var",
                "features.15.conv.0.1.weight",
                "features.15.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_180",
            "type": "Activation",
            "inputs": [
                "667"
            ],
            "outputs": [
                "667"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_181",
            "type": "Convolution2D",
            "inputs": [
                "667"
            ],
            "outputs": [
                "668"
            ],
            "params": [
                "features.15.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 960,
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
                "channels_out": 960,
                "channels_in": 960
            }
        },
        {
            "name": "BatchNormalization_182",
            "type": "BatchNorm",
            "inputs": [
                "668"
            ],
            "outputs": [
                "676"
            ],
            "params": [
                "features.15.conv.1.1.running_mean",
                "features.15.conv.1.1.running_var",
                "features.15.conv.1.1.weight",
                "features.15.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_185",
            "type": "Activation",
            "inputs": [
                "676"
            ],
            "outputs": [
                "676"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_186",
            "type": "Convolution2D",
            "inputs": [
                "676"
            ],
            "outputs": [
                "677"
            ],
            "params": [
                "features.15.conv.2.weight"
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
                "channels_out": 160,
                "channels_in": 960
            }
        },
        {
            "name": "BatchNormalization_187",
            "type": "BatchNorm",
            "inputs": [
                "677"
            ],
            "outputs": [
                "678"
            ],
            "params": [
                "features.15.conv.3.running_mean",
                "features.15.conv.3.running_var",
                "features.15.conv.3.weight",
                "features.15.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_188",
            "type": "Elementwise",
            "inputs": [
                "654",
                "678"
            ],
            "outputs": [
                "683"
            ],
            "options": {
                "operation": "sum"
            }
        },
        {
            "name": "Conv_189",
            "type": "Convolution2D",
            "inputs": [
                "683"
            ],
            "outputs": [
                "684"
            ],
            "params": [
                "features.16.conv.0.0.weight"
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
                "channels_out": 960,
                "channels_in": 160
            }
        },
        {
            "name": "BatchNormalization_190",
            "type": "BatchNorm",
            "inputs": [
                "684"
            ],
            "outputs": [
                "692"
            ],
            "params": [
                "features.16.conv.0.1.running_mean",
                "features.16.conv.0.1.running_var",
                "features.16.conv.0.1.weight",
                "features.16.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_193",
            "type": "Activation",
            "inputs": [
                "692"
            ],
            "outputs": [
                "692"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_194",
            "type": "Convolution2D",
            "inputs": [
                "692"
            ],
            "outputs": [
                "693"
            ],
            "params": [
                "features.16.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 960,
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
                "channels_out": 960,
                "channels_in": 960
            }
        },
        {
            "name": "BatchNormalization_195",
            "type": "BatchNorm",
            "inputs": [
                "693"
            ],
            "outputs": [
                "701"
            ],
            "params": [
                "features.16.conv.1.1.running_mean",
                "features.16.conv.1.1.running_var",
                "features.16.conv.1.1.weight",
                "features.16.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_198",
            "type": "Activation",
            "inputs": [
                "701"
            ],
            "outputs": [
                "701"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_199",
            "type": "Convolution2D",
            "inputs": [
                "701"
            ],
            "outputs": [
                "702"
            ],
            "params": [
                "features.16.conv.2.weight"
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
                "channels_out": 160,
                "channels_in": 960
            }
        },
        {
            "name": "BatchNormalization_200",
            "type": "BatchNorm",
            "inputs": [
                "702"
            ],
            "outputs": [
                "703"
            ],
            "params": [
                "features.16.conv.3.running_mean",
                "features.16.conv.3.running_var",
                "features.16.conv.3.weight",
                "features.16.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Add_201",
            "type": "Elementwise",
            "inputs": [
                "683",
                "703"
            ],
            "outputs": [
                "708"
            ],
            "options": {
                "operation": "sum"
            }
        },
        {
            "name": "Conv_202",
            "type": "Convolution2D",
            "inputs": [
                "708"
            ],
            "outputs": [
                "709"
            ],
            "params": [
                "features.17.conv.0.0.weight"
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
                "channels_out": 960,
                "channels_in": 160
            }
        },
        {
            "name": "BatchNormalization_203",
            "type": "BatchNorm",
            "inputs": [
                "709"
            ],
            "outputs": [
                "717"
            ],
            "params": [
                "features.17.conv.0.1.running_mean",
                "features.17.conv.0.1.running_var",
                "features.17.conv.0.1.weight",
                "features.17.conv.0.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_206",
            "type": "Activation",
            "inputs": [
                "717"
            ],
            "outputs": [
                "717"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_207",
            "type": "Convolution2D",
            "inputs": [
                "717"
            ],
            "outputs": [
                "718"
            ],
            "params": [
                "features.17.conv.1.0.weight"
            ],
            "options": {
                "bias": false,
                "groups": 960,
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
                "channels_out": 960,
                "channels_in": 960
            }
        },
        {
            "name": "BatchNormalization_208",
            "type": "BatchNorm",
            "inputs": [
                "718"
            ],
            "outputs": [
                "726"
            ],
            "params": [
                "features.17.conv.1.1.running_mean",
                "features.17.conv.1.1.running_var",
                "features.17.conv.1.1.weight",
                "features.17.conv.1.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_211",
            "type": "Activation",
            "inputs": [
                "726"
            ],
            "outputs": [
                "726"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "Conv_212",
            "type": "Convolution2D",
            "inputs": [
                "726"
            ],
            "outputs": [
                "727"
            ],
            "params": [
                "features.17.conv.2.weight"
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
                "channels_out": 320,
                "channels_in": 960
            }
        },
        {
            "name": "BatchNormalization_213",
            "type": "BatchNorm",
            "inputs": [
                "727"
            ],
            "outputs": [
                "728"
            ],
            "params": [
                "features.17.conv.3.running_mean",
                "features.17.conv.3.running_var",
                "features.17.conv.3.weight",
                "features.17.conv.3.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Conv_214",
            "type": "Convolution2D",
            "inputs": [
                "728"
            ],
            "outputs": [
                "733"
            ],
            "params": [
                "features.18.0.weight"
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
                "channels_out": 1280,
                "channels_in": 320
            }
        },
        {
            "name": "BatchNormalization_215",
            "type": "BatchNorm",
            "inputs": [
                "733"
            ],
            "outputs": [
                "741"
            ],
            "params": [
                "features.18.1.running_mean",
                "features.18.1.running_var",
                "features.18.1.weight",
                "features.18.1.bias"
            ],
            "options": {
                "eps": 9.999999747378752e-06,
                "momentum": 0.10000002384185791
            }
        },
        {
            "name": "Clip_218",
            "type": "Activation",
            "inputs": [
                "741"
            ],
            "outputs": [
                "741"
            ],
            "options": {
                "activation": "relu6"
            }
        },
        {
            "name": "GlobalAveragePool_219",
            "type": "GlobalPooling",
            "inputs": [
                "741"
            ],
            "outputs": [
                "753"
            ],
            "options": {
                "mode": "avg"
            }
        },
        {
            "name": "Gemm_231",
            "type": "InnerProduct",
            "inputs": [
                "753"
            ],
            "outputs": [
                "loss"
            ],
            "params": [
                "classifier.1.weight",
                "classifier.1.bias"
            ],
            "options": {
                "bias": true,
                "outputs": 1000,
                "inputs": 1280
            }
        }
    ]
}