#include <dlprim/tensor.hpp>

#include <iostream>

namespace dlprim {
	struct Tensor::HostMem {
		void *p = nullptr;
		
		~HostMem()
		{
			free();
		}
		void free()
		{
			if(p) {
			#ifndef DLPRIM_WINDOWS
				free(p); 
			#else
				_aligned_free(p);
			#endif
			}
			p = nullptr;
		}
		void alloc(size_t size)
		{
			free();
            #ifndef DLPRIM_WINDOWS
			p = aligned_alloc(128,size);
            #else
            p = _aligned_malloc(size,128);
            #endif
			if(!p)
				throw std::bad_alloc();
		}
		
	};
    Tensor::Tensor() :
       host_(new Tensor::HostMem()),
       cpu_tensor_(true),
	   offset_(0)
    {
    }
    Tensor::Tensor(Context &ctx,Shape const &s,DataType d):
        TensorSpecs(s,d),
		host_(new Tensor::HostMem()),
        cpu_tensor_(ctx.is_cpu_context()),
        offset_(0),
        capacity_(s.total_size())
    {
        size_t size = memory_size();
        DLPRIM_CHECK(size > 0);
		if(cpu_tensor_)
			host_->alloc(size);
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
        c.queue().enqueueWriteBuffer(buffer_, sync ? CL_TRUE : CL_FALSE, 0, memory_size(), host_data(),c.events(),c.event("write"));
    }
    void Tensor::to_host(ExecutionContext const &c,bool sync)
    {
        if(cpu_tensor_)
            return;
        c.queue().enqueueReadBuffer(buffer_, sync ? CL_TRUE : CL_FALSE, 0, memory_size(), host_data(),c.events(),c.event("read"));
    }

    void *Tensor::host_data()
    {
		if(!host_->p) {
			host_->alloc(memory_size());
		}
        return host_->p;
    }
};
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
