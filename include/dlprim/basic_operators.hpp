#pragma once
#include <dlprim/tensor.hpp>
namespace dlprim {
    namespace ops {

        class OclOp {
        public:
            OclOp(Context const &ctx);
            void build(char const *kernel,char const *name)
            {
                cl::Program::Sources src(1,std::make_pair(kernel,strlen(kernel)));
                program_ = cl::Program(ctx_.context(),src);
                program_.build(ctx_.device());
                kernel_ = cl::Kernel(program_,name);
            }
        protected:
            cl::Kernel kernel_;
            cl::Program progam_;
            Context ctx_;
        
            ///
        };

        class ReLU : public OclOp {
        public:
            ReLU(Context const &ctx,DataType dt=float_data)  : OclOp(ctx)
            {
                build(R"xxx(
                __kernel
                __attribute__((reqd_work_group_size(256,1,1))
                void relu(int N, __global const Dtype *in,int in_off,__global Dtype *out,int out_off)
                {
                    in += in_off;
                    out += out_off;
                    int pos = get_global_id(0);
                    if(pos >= N)
                        return;
                    out[pos] = max(in[pos],0);
                }
                )xxx","relu",get_type_name(d));
            }
        };
        class GEMM : public OclOp {
        public:
            GEMM(Context const &ctx,DataType dtype=float_data);
            ~GEMM();
            void operator()(bool tra,bool trb,Tensor &a,Tensor &b,Tensor &c,float alpha=1.0,float beta=0.0,cl::CommandQueue &q);
        };

        class ReLU {
        };

    };

