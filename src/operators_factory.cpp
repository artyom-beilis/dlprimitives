///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/operator.hpp>
#include <dlprim/functions.hpp>
#include <dlprim/operators.hpp>
#include <dlprim/json.hpp>
#include <functional>

namespace dlprim {
    
static std::map<std::string,std::function<Operator *(Context &,json::value const &)> > generators = {
    { 
        "Softmax", 
        [](Context &ctx,json::value const &p) {
            return new Softmax(ctx,SoftmaxConfig::from_json(p));
        }
    },
    { 
        "NLLLoss", 
        [](Context &ctx,json::value const &p) {
            return new NLLLoss(ctx,NLLLossConfig::from_json(p));
        }
    },
    { 
        "MSELoss", 
        [](Context &ctx,json::value const &p) {
            return new MSELoss(ctx,MSELossConfig::from_json(p));
        }
    },
    { 
        "SoftmaxWithLoss", 
        [](Context &ctx,json::value const &p) {
            return new SoftmaxWithLoss(ctx,SoftmaxConfig::from_json(p));
        }
    },
    {
        "Activation", 
        [](Context &ctx,json::value const &p) {
            return new Activation(ctx,ActivationConfig::from_json(p));
        }
    },
    {
        "Elementwise", 
        [](Context &ctx,json::value const &p) {
            return new Elementwise(ctx,ElementwiseConfig::from_json(p));
        }
    },
    {
        "Pooling2D", 
        [](Context &ctx,json::value const &p) {
            return new Pooling2D(ctx,Pooling2DConfig::from_json(p));
        }
    },
    {
        "GlobalPooling", 
        [](Context &ctx,json::value const &p) {
            return new GlobalPooling(ctx,GlobalPoolingConfig::from_json(p));
        }
    },
    {
        "InnerProduct", 
        [](Context &ctx,json::value const &p) {
            return new InnerProduct(ctx,InnerProductConfig::from_json(p));
        }
    },
    {
        "Convolution2D", 
        [](Context &ctx,json::value const &p) {
            return new Convolution2D(ctx,Convolution2DConfig::from_json(p));
        }
    },
    {
        "TransposedConvolution2D", 
        [](Context &ctx,json::value const &p) {
            return new TransposedConvolution2D(ctx,TransposedConvolution2DConfig::from_json(p));
        }
    },
    {
        "BatchNorm", 
        [](Context &ctx,json::value const &p) {
            return new BatchNorm(ctx,BatchNormConfig::from_json(p));
        }
    },
    {
        "Flatten", 
        [](Context &ctx,json::value const &p) {
            return new Flatten(ctx,FlattenConfig::from_json(p));
        }
    },
    {
        "Squeeze", 
        [](Context &ctx,json::value const &p) {
            return new Squeeze(ctx,SqueezeConfig::from_json(p));
        }
    },
    {
        "Reshape", 
        [](Context &ctx,json::value const &p) {
            return new Reshape(ctx,ReshapeConfig::from_json(p));
        }
    },
    {
        "Concat", 
        [](Context &ctx,json::value const &p) {
            return new Concat(ctx,ConcatConfig::from_json(p));
        }
    },
    {
        "Slice", 
        [](Context &ctx,json::value const &p) {
            return new Slice(ctx,SliceConfig::from_json(p));
        }
    },
    {
        "Threshold", 
        [](Context &ctx,json::value const &p) {
            return new Threshold(ctx,ThresholdConfig::from_json(p));
        }
    },
    {
        "Abs", 
        [](Context &ctx,json::value const &p) {
            return new Abs(ctx,AbsConfig::from_json(p));
        }
    },
    {
        "Hardtanh", 
        [](Context &ctx,json::value const &p) {
            return new Hardtanh(ctx,HardtanhConfig::from_json(p));
        }
    },
    {
        "Reduction", 
        [](Context &ctx,json::value const &p) {
            return new Reduction(ctx,ReductionConfig::from_json(p));
        }
    },
    {
        "Parameter", 
        [](Context &ctx,json::value const &p) {
            return new Parameter(ctx,ParameterConfig::from_json(p));
        }
    },
    {
        "Interpolation", 
        [](Context &ctx,json::value const &p) {
            return new Interpolation(ctx,InterpolationConfig::from_json(p));
        }
    },
};
    
std::unique_ptr<Operator> create_by_name(Context &ctx,
                                        std::string const &name,
                                        json::value const &parameters)
{
    auto p=generators.find(name);
    if(p == generators.end()) {
        throw ValidationError("Unknown operator " + name);
    }
    std::unique_ptr<Operator> r(p->second(ctx,parameters));
    return r;

}

} /// namespace
