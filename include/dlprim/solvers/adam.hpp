///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/net.hpp>
#include <dlprim/ops/scal.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/ops/initialization.hpp>
#include <dlprim/solvers/solver_base.hpp>
#include <iostream>
#include <cmath>

namespace dlprim {
    namespace solvers {
        class Adam : public SolverBase {
        public:
            float lr    = 0.001;
            float beta1 = 0.9;
            float beta2 = 0.999;
            float eps   = 1e-8;
            float weight_decay = 0.0005; 
            Adam(Context &ctx) : ctx_(ctx)
            {
            }
            void init(Net &n,ExecutionContext const &q)
            {
                for(auto &p : n.param_diffs()) {
                    auto &v = v_[p.first] = Tensor(ctx_,p.second.shape(),p.second.dtype());
                    auto &m = m_[p.first] = Tensor(ctx_,p.second.shape(),p.second.dtype());
                    set_to_zero(v,q);
                    set_to_zero(m,q);
                }
                t_ = 0;
            }
            void zero_grad(Net &n,ExecutionContext const &e)
            {
                for(auto &p : n.param_diffs()) {
                    set_to_zero(p.second,e);
                }
            }
            void apply(Net &n,ExecutionContext const &e)
            {
                t_++;
                inv_b1_ = 1 / (1 - std::pow(beta1,t_));
                inv_b2_ = 1 / (1 - std::pow(beta2,t_));

                for(auto &item : n.param_diffs()) {
                    std::string const &name = item.first;
                    Tensor &v = v_[name];
                    Tensor &p = n.param(name);
                    Tensor &g = item.second;
                    Tensor &m = m_[name];
                    if(ctx_.is_cpu_context()) {
                        apply_cpu(p,g,m,v);
                    }
                    else {
                        apply_gpu(p,g,m,v,e);
                    }
               }
            }
        private:
            void apply_cpu(Tensor &p_t,Tensor &g_t,Tensor &m_t,Tensor &v_t)
            {
                size_t size = p_t.shape().total_size();
                float *p = p_t.data<float>();
                float *g = g_t.data<float>();
                float *m = m_t.data<float>();
                float *v = v_t.data<float>();
                for(size_t i=0;i<size;i++) {
                    float grad = g[i] + weight_decay * p[i];
                    float m_next = beta1 * m[i] + (1-beta1) * grad;
                    float v_next = beta2 * v[i] + (1-beta2) * grad * grad;
                    float m_top = m_next * inv_b1_;
                    float v_top = v_next * inv_b2_;
                    float p_next = p[i] - lr * m_top / (std::sqrt(v_top) + eps);

                    m[i] = m_next;
                    v[i] = v_next;
                    p[i] = p_next;
                }
            }
            void apply_gpu(Tensor &p,Tensor &g,Tensor &m,Tensor &v,ExecutionContext const &e)
            {
                core::pointwise_operation({p,g,m,v},
                                          {p,m,v},
                                          {beta1,beta2,inv_b1_,inv_b2_,lr,weight_decay,eps},
                                          R"xxx(
                                            dtype p=x0, g=x1, m=x2, v=x3;
                                            dtype beta1 = w0,beta2 = w1,inv_b1 = w2,inv_b2=w3,lr=w4,weight_decay=w5,eps=w6;
                                            dtype grad = g + weight_decay * p;
                                            dtype m_next = beta1 * m + (1-beta1) * grad;
                                            dtype v_next = beta2 * v + (1-beta2) * grad * grad;
                                            dtype m_top = m_next * inv_b1;
                                            dtype v_top = v_next * inv_b2;
                                            dtype p_next = p - lr * m_top / (sqrt(v_top) + eps);
                                            y0 = p_next;
                                            y1 = m_next;
                                            y2 = v_next;
                                          )xxx",
                    e);
            }

            Context ctx_;
            std::map<std::string,Tensor> m_;
            std::map<std::string,Tensor> v_;
            int t_;
            float inv_b1_,inv_b2_;
        };
    } // solvers
}
