#include <dlprim/tensor.hpp>

namespace dlprim {
    Tensor::Tensor() :
       cpu_tensor_(true),
       offset_(0)
    {
    }
    Tensor::Tensor(Context &ctx,Shape const &s,DataType d):
        TensorSpecs(s,d),
        cpu_tensor_(ctx.is_cpu_context()),
        offset_(0)
    {
        size_t size = memory_size();
        DLPRIM_CHECK(size > 0);
        if(cpu_tensor_) {
            void *ptr = 
            #ifndef DLPRIM_WINDOWS
              aligned_alloc(128,size);
            #else
              _aligned_malloc(size,128);
            #endif
            if(!ptr)
                throw std::bad_alloc();
            host_ = std::shared_ptr<void>(ptr,[](void *p) { free(p); });
        }
        else {
            buffer_ = cl::Buffer(ctx.context(),CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,size);
            cl::CommandQueue q(ctx.context());
            void *ptr = q.enqueueMapBuffer(buffer_,CL_TRUE,0,0,size);
            host_ = std::shared_ptr<void>(ptr,[](void *) {});
        }
    }
    void Tensor::to_device(cl::CommandQueue &q,bool sync)
    {
        q.enqueueWriteBuffer(buffer_, sync ? CL_TRUE : CL_FALSE, 0, memory_size(), host_.get());
    }
    void Tensor::to_host(cl::CommandQueue &q,bool sync)
    {
        q.enqueueReadBuffer(buffer_, sync ? CL_TRUE : CL_FALSE, 0, memory_size(), host_.get());
    }

    void *Tensor::host_data()
    {
        DLPRIM_CHECK(!cpu_tensor_);
        return host_.get();
    }
};
