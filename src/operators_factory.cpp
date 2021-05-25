#include <dlprim/operator.hpp>
#include <dlprim/functions.hpp>
#include <dlprim/operators.hpp>
#include <dlprim/json.hpp>
#include <functional>

namespace dlprim {
    
static std::map<std::string,std::function<Operator *(Context &,json::value const &)> > generators = {
    { 
        "SoftMax", 
        [](Context &ctx,json::value const &p) {
            return new SoftMax(ctx,SoftMaxConfig::from_json(p));
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
    }
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
