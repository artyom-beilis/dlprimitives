///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/operator.hpp>
namespace dlprim {
    namespace core { class BiasBackwardFilter; }
    class BWBias {
    public:
        BWBias(Context &ctx,Shape const &sp,DataType dt=float_data);
        ~BWBias();
        void backward(Tensor &dy,Tensor &dw,Tensor &ws,float beta,ExecutionContext const &e);
        size_t workspace() const;
    private:
        void backward_cpu(Tensor &dy,Tensor &dw,float beta);
        std::unique_ptr<core::BiasBackwardFilter> impl_;
    };
}
