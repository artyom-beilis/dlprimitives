///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/net.hpp>
#include <iostream>
#include <dlprim/ops/scal.hpp>
#include <dlprim/ops/axpby.hpp>
#include <dlprim/solvers/solver_base.hpp>
namespace dlprim {
    namespace solvers {
        class SGD : public SolverBase {
        public:
            float lr = 0.1;
            float momentum = 0.9;
            float weight_decay = 0.0005; 
            SGD(Context &ctx,DataType dtype = float_data) : 
                ctx_(ctx),
                scal_(ctx,dtype),axpby_(ctx,dtype) 
            {
            }
            void init(Net &n,ExecutionContext const &q)
            {
                for(auto &p : n.param_diffs()) {
                    auto &t = vel_[p.first] = Tensor(ctx_,p.second.shape(),p.second.dtype());
                    scal_.scale(0,t,q);
                }
            }
            void zero_grad(Net &n,ExecutionContext const &e)
            {
                for(auto &p : n.param_diffs()) {
                    scal_.scale(0,p.second,e);
                }
            }
            void apply(Net &n,ExecutionContext const &e)
            {
                for(auto &item : vel_) {
                    std::string const &name = item.first;
                    Tensor &v = item.second;
                    Tensor &p = n.param(name);
                    Tensor &g = n.param_diff(name);
                    axpby_.apply(1.0,g,momentum,v,v,e);  // v = momentum * v - lr * gr
                    axpby_.apply((1.0f-weight_decay),p,-lr,v,p,e);
               }
            }
        private:
            Context ctx_;
            Scal scal_;
            AXPBY axpby_;

            std::map<std::string,Tensor> vel_;
        };
    } // solvers
}
