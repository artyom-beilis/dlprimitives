#pragma once
#include <dlprim/tensor.hpp>
#include <dlprim/context.hpp>
namespace dlprim {
namespace core {

    ///
    /// per form operations function(xs,ws)->yw such that
    /// each tensor in xs and ys has same shape, ws are constant parameters
    ///
    /// code should perform assignment to variables y0 to yN and use x0..xM as values, and w0...wK as parameters
    /// for example:
    ///   `pointwise_operation({a,b},{c},{w},"y0 = x0*w0 + x1;",q);`
    ///
    void pointwise_operation(std::vector<Tensor> xs,
                             std::vector<Tensor> ys,
                             std::vector<double>  ws,
                             std::string const &code,
                             ExecutionContext const &ec);

} // core
} // dlprim
