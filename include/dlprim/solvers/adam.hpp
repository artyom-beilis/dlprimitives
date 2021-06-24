#include <dlprim/net.hpp>
#include <dlprim/ops/scal.hpp>
#include <iostream>
#include <cmath>

namespace dlprim {
    namespace solvers {
        class Adam {
        public:
            float lr    = 0.001;
            float beta1 = 0.9;
            float beta2 = 0.999;
            float eps   = 1e-8;
            float weight_decay = 0.0005; 
            Adam(Context &ctx,DataType dtype = float_data) : 
                ctx_(ctx),
                scal_(ctx,dtype)
            {
                if(ctx_.is_gpu_context()) {
                    cl::Program const &prog = gpu::Cache::instance().get_program(ctx_,"adam");
                    adam_ = cl::Kernel(prog,"adam");
                }
            }
            void init(Net &n,ExecutionContext const &q)
            {
                for(auto &p : n.param_diffs()) {
                    auto &v = v_[p.first] = Tensor(ctx_,p.second.shape(),p.second.dtype());
                    auto &m = m_[p.first] = Tensor(ctx_,p.second.shape(),p.second.dtype());
                    scal_.scale(0,v,q);
                    scal_.scale(0,m,q);
                }
                t_ = 0;
            }
            void zero_grad(Net &n,ExecutionContext const &e)
            {
                for(auto &p : n.param_diffs()) {
                    scal_.scale(0,p.second,e);
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
            void apply_gpu(Tensor &pr,Tensor &g,Tensor &m,Tensor &v,ExecutionContext const &e)
            {
                int size = pr.shape().total_size();
                int p=0;
                adam_.setArg(p++,size);
                adam_.setArg(p++,beta1);
                adam_.setArg(p++,beta2);
                adam_.setArg(p++,inv_b1_);
                adam_.setArg(p++,inv_b2_);
                adam_.setArg(p++,lr);
                adam_.setArg(p++,weight_decay);
                adam_.setArg(p++,eps);
                adam_.setArg(p++,pr.device_buffer());
                adam_.setArg(p++,int(pr.device_offset()));
                adam_.setArg(p++,g.device_buffer());
                adam_.setArg(p++,int(g.device_offset()));
                adam_.setArg(p++,m.device_buffer());
                adam_.setArg(p++,int(m.device_offset()));
                adam_.setArg(p++,v.device_buffer());
                adam_.setArg(p++,int(v.device_offset()));
                e.queue().enqueueNDRangeKernel(adam_,cl::NullRange,cl::NDRange(size),cl::NullRange,e.events(),e.event("adam"));
            }

            Context ctx_;
            Scal scal_;
            cl::Kernel adam_;

            std::map<std::string,Tensor> m_;
            std::map<std::string,Tensor> v_;
            int t_;
            float inv_b1_,inv_b2_;
        };
    } // solvers
}
