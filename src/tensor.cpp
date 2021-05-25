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
        offset_(0),
        capacity_(s.total_size())
    {
        size_t size = memory_size();
        DLPRIM_CHECK(size > 0);
        //if(cpu_tensor_) {
            void *ptr = 
            #ifndef DLPRIM_WINDOWS
              aligned_alloc(128,size);
            #else
              _aligned_malloc(size,128);
            #endif
            if(!ptr)
                throw std::bad_alloc();
            host_ = std::shared_ptr<void>(ptr,[](void *p) { free(p); });
        //}
        if(!cpu_tensor_) {
            buffer_ = cl::Buffer(ctx.context(),CL_MEM_READ_WRITE,size);
        }
    }

    void Tensor::reshape(Shape const &new_shape)
    {
        if(new_shape.total_size() > capacity_)
            throw ValidationError("respae: new size is larger than original");
        shape_ = new_shape;
    }

    void Tensor::to_device(ExecutionContext const &c,bool sync)
    {
        if(cpu_tensor_)
            return;
        c.queue().enqueueWriteBuffer(buffer_, sync ? CL_TRUE : CL_FALSE, 0, memory_size(), host_.get(),c.events(),c.event());
    }
    void Tensor::to_host(ExecutionContext const &c,bool sync)
    {
        if(cpu_tensor_)
            return;
        c.queue().enqueueReadBuffer(buffer_, sync ? CL_TRUE : CL_FALSE, 0, memory_size(), host_.get(),c.events(),c.event());
    }

    void *Tensor::host_data()
    {
        DLPRIM_CHECK(host_.get());
        return host_.get();
    }
};
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
