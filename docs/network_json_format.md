# Network JSON Format

JSON Network format consists of three field, each one of them is array

- `inputs` - list of network input tensors
- `outputs` - list of network output tensors
- `operators` - list of operators executed in graph

## Inputs

Input is object that carries following fields:

- `name` - tensor name string
- `shape` - tensor shape list of integers
- `dtype` - type of tensor string, default is "float"

For example:

    "inputs": [
        {
            "shape": [ 64, 1, 28, 28 ],
            "name":  "data"
        },
        {
            "shape": [ 64 ],
            "name":  "label",
            "dtype": "int"
        }
    ],


## Outputs

Outputs are either string representing output tensor name or an object with fields `name` with string value as tensor name and `loss_weight` for weight for loss.

Note: if tensor name starts with `loss` it is considered a loss and back propagation goes from it. Default loss weight is 1.0. If `loss_weight` is provided than the tensor considered as loss and it will participate in back propagation.

For example

    "outputs" : [ 
        "prob", 
        { "name" : "cross_entropy_loss", "loss_weight": 1.0 }
    ]

## Operators

Operators is list of operators executed during forward and back-propagation in the order provided. They create edges between nodes. The graph must be acyclic with exception of in-place operators (i.e. it may be self edge)

Each operator has following fields:

- `name` - unique name of operator, string
- `type` - type of operation, string, for example "Activation"
- `inputs` - list of input tensors - must be either outputs of one of previous operators or one of inputs - list of strings
- `outputs` - list of generated tensors, can be same as input for in-place operations - list of strings. 
- `frozen` - boolean flag that marks the operator as not-participating in gradient descent, for example for transfer learning. Default is `false`
- `options` - object operation specific parameters
- `params` - list of strings, optional non standard name of parameters for imported data/parameter sharing

For example:

    {
        "name": "fc2",
        "type": "InnerProduct",
        "inputs": ["fc1"],
        "outputs": ["fc2"],
        "options": {"outputs": 10}
    }

## Supported Operators and Options

### SoftmaxWithLoss

No parameters at this point

### NLLLoss

Negative Likelihood Log Loss. Expects log of probability as input.

Parameters:

- `reduce` - reduction, default `mean`, one of `none`, `mean` or `sum` - the reduction on output values.

### MSELoss

Mean Square Error loss, expects as input two identical shapes

Parameters:

- `reduce` - reduction, default `mean`, one of `none`, `mean` or `sum` - the reduction on output values.



### Softmax

Parameters:

- `log` - boolean, default false. Output a log of the softmax value rather than original value, better for numerical stability


### Activation

Parameters:

- `activation` - string, one of standard activation names, relu, tanh, sigmoid, relu6, identity

### Elementwise


Parameters:

- `operation` - string, one of `sum` for `ax + by`, `prod` - for `ax * by` , `max` for `max(ax,by)` where, `x` and `y` are input tensors, `a` and `b` are coefficients - default is `sum` 
- `coef1` - scale for 1st input, number default 1
- `coef2` - scale for 2nd input, number default 1
- `activation` - string one of standard activation names

### Pooling2D

Parameters

- `mode` - string, one of `max` and `avg`, default is `max`
- `kernel` - integer or list of two integers, pooling kernel size.
- `stride` - integer or list of two integers, pooling stride, default 1
- `pad`  - integer or list of two integer, padding, default 0
- `count_include_pad` - boolean calculate average over padded area as well, default false
- `ceil_mode` - boolean, default false, how to round the strided pooling size upwards or downwards

Note: kernel, stride and pad can be either single number for symmetric values or pair of integers, 1st for height dimension and second for width

### GlobalPooling

Parameters

- `mode` - string, one of `max` and `avg`, default is `max`

### InnerProduct

Parameters

- `outputs` - integer - number of output features
- `inputs` - integer - number of input features, can be deduced automatically
- `bias` - boolean - apply bias, default true
- `activation` - string, optional, one of standard activation names

### Convolution2D

Parameters:

- `channels_out` - integer, number of output channels/features
- `channels_in` - number of input channels, can be deduced automatically
- `groups` - number of convolution groups, integer, default 1
- `bias` - boolean - apply bias, default true
- `kernel` - integer or list of two integers, convolution kernel size.
- `stride` - integer or list of two integers, convolution stride, default 1
- `pad`  - integer or list of two integer, padding, default 0
- `dilate`  - integer or list of two integer, dilation, default 1

Note: kernel, stride, dilate and pad can be either single number for symmetric values or pair of integers, 1st for height dimension and second for width

### TransposedConvolution2D

Parameters:

- `channels_out` - integer, number of output channels/features
- `channels_in` - number of input channels, can be deduced automatically
- `groups` - number of convolution groups, integer, default 1
- `bias` - boolean - apply bias, default true
- `kernel` - integer or list of two integers, convolution kernel size.
- `stride` - integer or list of two integers, convolution stride, default 1
- `pad`  - integer or list of two integer, padding, default 0
- `output_pad`  - integer or list of two integer, padding of output in order to solve ambiguity if "input" size due to strides, default 0
- `dilate`  - integer or list of two integer, dilation, default 1

Note: kernel, stride, dilate, pad and output\_pad can be either single number for symmetric values or pair of integers, 1st for height dimension and second for width


### BatchNorm

Parameters

- `features` - integer number of input features, can be automatically deduced
- `eps` - floating point value, default 1e-5, epsilon for batch normalization `y = x / sqrt( var + eps)`
- `momentum` - floating point value, default 0.1, portion of the newly calculated mean/variance in running sums: 
  `running_sum := (1-momentum)*running_sum + momentum * batch_sum`
- `affine` - boolean, default true, apply additional trainable gamma scale and beta offset after normalization
- `use_global_stats` - use previously calculated mean/variance instead of calculating them per-batch and updating running sums. Useful for freezing layer, default false. Note: for testing it is "always true"

### Concat

Parameters

- `dim` - concatenate input tensors over dimension dim, default 1

### Slice

Parameters

- `dim` - slice input tensor over dimension dim, default 1
- `begin` - begin index of slice, default 0
- `end` - end index of slice, default end 

For example: `{ "begin":1, "end":2","dim":1 }` - slice green channel

### Flatten

Flattens the shape to `[batch,features]` no parameters

### Squeeze

Squeezes the shape

- `all` default true if dims empty otherwise false, squeeze all 1 dimensions
- `dims` - list dimension to squeeze, negative counted from end

### Reshape

Reshapes the tensor

- `dims` - list of ints new dimension, 0 - keep same as origin, -1 deduce from others


### Reduction

Performs reduction over different axes

Parameters:

- `method` - string, default `sum`. Options `sum`: sum, `sumsq`: sum of squares, `abssum`: L1 norm, `mean` - average value
- `keep_dim` - boolean, default true, whether to keep dimension. For result in reduction of shape [2,3,4] over axes 1,2, the output will have shape [2,1,1] of `keep_dim` is true and otherwise it will be [2]
- `output_scale` - float, default 1, multiply result after reduction by factor
- `dims` - list of ints, dimensions to reduce over, may be negative, can't be specified together with `start_axis`, 
- `start_axis` - integer, default 0 - first axis to reduce till end, for example if `start_axis==1` and input shape is [2,3,4], result shape will be [2,1,1]. Can't specify both `start_axis`and `dims`

### Threshold

Compute `x > threshold ? 1 : 0`

Parameters

- `threshold`, default 0

### Hardtanh

Compute `max(min_val,min(max_val,x))`

Parameters

- `min_val`, default -1
- `max_val , default 1


### Abs

Compute `abs(x)`, no parameters


### Parameter

Special layer with no inputs that produces output of its only paramerer

- `shape` - list of ints - output shape
- `dtype` - string, default float - type of output tensor
- `is_trainable` - boolean default true, backpropagate gradients to parameter


## Standard Activations

Following are standard activation names: `relu`, `sigmoid`, `tanh`, `relu6`, `identity`

## Example MNIST MLP Network

This is an example of a simple MLP network for mnist training

    {
        "inputs": [
            { 
                "shape": [ 64,1,28,28 ],
                "name": "data"
            },
            {
                "shape": [64],
                "name": "label",
                "dtype" : "int"
            }
        ],
        "outputs" : [ 
            "prob", "loss" 
        ],
        "operators": [
            {
                "name": "fc1",
                "type": "InnerProduct",
                "inputs": [ "data" ],
                "outputs": [ "fc1" ],
                "options": {
                    "outputs": 256,
                    "activation": "relu"
                }
            },
            {
                "name": "fc2",
                "type": "InnerProduct",
                "inputs": ["fc1" ],
                "outputs": [ "fc2" ],
                "options": {
                    "outputs": 10
                }
            },
            {
                "name": "prob",
                "type": "Softmax",
                "inputs": ["fc2"],
                "outputs": ["prob"]
            },
            {
                "name": "loss",
                "type": "SoftmaxWithLoss",
                "inputs": ["fc2","label"],
                "outputs": ["loss"]
            }
        ]
    }
