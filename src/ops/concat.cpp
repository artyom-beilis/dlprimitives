#include <dlprim/ops/concat.hpp>
#include <dlprim/core_ops.hpp>
#include <dlprim/json.hpp>
#include <my_cblas.hpp>
namespace dlprim {
    ConcatConfig ConcatConfig::from_json(json::value const &v)
    {
        ConcatConfig cfg;
        cfg.dim = v.get<int>("dim",cfg.dim);
        return cfg;
    }
    Concat::Concat(Context &ctx,ConcatConfig const &config) : Operator(ctx), cfg_(config)
    {
    }
    Concat::~Concat()
    {
    }
	void Concat::setup(std::vector<TensorSpecs> const &in,
                       std::vector<TensorSpecs> &out,
                       std::vector<TensorSpecs> &parameters,
                       size_t &workspace)
    {
        std::vector<Shape> ins,outs;
        dtype_ = in.at(0).dtype();
        DLPRIM_CHECK(dtype_==float_data);
        for(size_t i=0;i<in.size();i++) {
            ins.push_back(in[i].shape());
            DLPRIM_CHECK(in[0].dtype() == in[i].dtype());
        }
        reshape(ins,outs,workspace);
        out.clear();
        out.push_back(TensorSpecs(outs[0],in[0].dtype()));
        parameters.clear();
        if(ctx_.is_opencl_context()) {
            copy_.reset(new core::SliceCopy(ctx_,dtype_));
        }
    }
    void Concat::reshape(std::vector<Shape> const &in,
                         std::vector<Shape> &out,
                         size_t &ws)
    {
        size_t total_dim = 0;
        for(size_t i=0;i<in.size();i++) {
            DLPRIM_CHECK(cfg_.dim < in[i].size());
            total_dim += in[i][cfg_.dim];
            if(i == 0)
                continue;
            DLPRIM_CHECK(in[0].size() == in[i].size());
            for(int j=0;j<in[0].size();j++) {
                if(j==cfg_.dim)
                    continue;
                DLPRIM_CHECK(in[0][j] == in[i][j]);
            }
        }
        Shape r = in[0];
        r[cfg_.dim] = total_dim;
        out.assign({r});
        ws = 0;
    }
    template<typename Cp>
    void Concat::copy_cpu(size_t &offset,Tensor &in,Tensor &out,Cp copy)
    {
        Shape target = out.shape().split_and_merge_over_axis(cfg_.dim);
        Shape cur = in.shape().split_and_merge_over_axis(cfg_.dim);
        DLPRIM_CHECK(cur[0]==target[0]);
        DLPRIM_CHECK(cur[2]==target[2]);
        DLPRIM_CHECK(cur[1]+offset <= target[1]);
        for(size_t d0 = 0; d0 < cur[0]; d0++) {
            for(size_t d1=0;d1 < cur[1];d1++) {
                size_t src_offset = d1 * cur[2] + d0 * (cur[2]*cur[1]);
                size_t tgt_offset = (d1 + offset) * target[2] + d0 * (target[2]*target[1]);
                float *src = in.data<float>()  + src_offset;
                float *tgt = out.data<float>() + tgt_offset;
                copy(tgt,src,cur[2]);
            }
        }
        offset += cur[1];
    }
    
    struct FwdCopyBlock {
        void operator()(float *tgt,float *src,size_t len)
        {
            memcpy(tgt,src,sizeof(float)*len);
        }
    };

    struct FwdCopyOne {
        void operator()(float *tgt,float *src,size_t )
        {
            *tgt=*src;
        }
    };


    struct BwdCopyBlock {
        void operator()(float *tgt,float *src,size_t len)
        {
            memcpy(src,tgt,len*sizeof(float));
        }
    };

    struct BwdCopyBlockFactor {
        float f;
        BwdCopyBlockFactor(float v) : f(v) {}
        void operator()(float *tgt,float *src,size_t len)
        {
            cblas_sscal(len,f,src,1);
            cblas_saxpy(len,1.0f,tgt,1,src,1);
        }
    };
    struct BwdCopyOne {
        void operator()(float *tgt,float *src,size_t)
        {
            *src = *tgt;
        }
    };
    struct BwdCopyOneFactor {
        float f;
        BwdCopyOneFactor(float v) :f(v) {}
        void operator()(float *tgt,float *src,size_t)
        {
            *src = *src *f + *tgt;
        }
    };


	void Concat::forward(std::vector<Tensor> &input,
                         std::vector<Tensor> &output,
                         std::vector<Tensor> &,
                         Tensor &,
                         ExecutionContext const &q)
    {
        DLPRIM_CHECK(output.size()==1);
        bool copy_one = input.at(0).shape().split_and_merge_over_axis(cfg_.dim)[2] == 1;
        size_t offset = 0;
        for(size_t i=0;i<input.size();i++) {
            if(ctx_.is_cpu_context()) {
                if(copy_one)
                    copy_cpu(offset,input[i],output[0],FwdCopyOne());
                else
                    copy_cpu(offset,input[i],output[0],FwdCopyBlock());
            }
            else {
                size_t slice = input[i].shape()[cfg_.dim];
                copy_->tensor_slice_copy(cfg_.dim,slice,
                                         output[0],offset,
                                         input[i],0,
                                         0.0f,q.generate_series_context(i,input.size()));
                offset += slice;
            }
        }
        DLPRIM_CHECK(offset == output[0].shape()[cfg_.dim]);
    }
    void Concat::backward(std::vector<TensorAndGradient> &input,
                          std::vector<TensorAndGradient> &output,
                          std::vector<TensorAndGradient> &,
                          Tensor &,
                          ExecutionContext const &q)
    {
        DLPRIM_CHECK(output.size()==1);
        bool copy_one = output.at(0).diff.shape().split_and_merge_over_axis(cfg_.dim)[2] == 1;
        size_t offset = 0;
        int total = 0;
        int count = 0;
        for(auto const &inp : input)
            total += int(inp.requires_gradient);

        for(size_t i=0;i<input.size();i++)  {
            if(!input[i].requires_gradient) {
                offset += input[i].data.shape()[cfg_.dim];
                continue;
            }
            if(ctx_.is_opencl_context()) {
                size_t slice = input[i].diff.shape()[cfg_.dim];
                copy_->tensor_slice_copy(cfg_.dim,slice,
                                         input[i].diff,0,
                                         output[0].diff,offset,
                                         input[i].accumulate_gradient,
                                         q.generate_series_context(count++,total));
                offset += slice;
            }
            else {
                if(copy_one) {
                    if(input[i].accumulate_gradient == 0)
                        copy_cpu(offset,input[i].diff,output[0].diff,BwdCopyOne());
                    else
                        copy_cpu(offset,input[i].diff,output[0].diff,BwdCopyOneFactor(input[i].accumulate_gradient));
                }
                else {
                    if(input[i].accumulate_gradient == 0)
                        copy_cpu(offset,input[i].diff,output[0].diff,BwdCopyBlock());
                    else
                        copy_cpu(offset,input[i].diff,output[0].diff,BwdCopyBlockFactor(input[i].accumulate_gradient));
                }
            }
        }
        DLPRIM_CHECK(offset == output[0].diff.shape()[cfg_.dim]);
    }
    
} // dlprim
