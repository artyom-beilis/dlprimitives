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
				::free(p); 
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
            int r = posix_memalign(&p,128,size);
            if(r!=0) {
                free();
                throw std::bad_alloc();
            }
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
	   offset_(0),
       capacity_(0),full_capacity_(0)
    {
    }
    Tensor::Tensor(Context &ctx,Shape const &s,DataType d,bool is_train):
        TensorSpecs(s,d,is_train),
		host_(new Tensor::HostMem()),
        cpu_tensor_(ctx.is_cpu_context()),
        offset_(0),
        capacity_(s.total_size()*size_of_data_type(d)),
        full_capacity_(capacity_)
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
        c.queue().enqueueWriteBuffer(buffer_, sync ? CL_TRUE : CL_FALSE, offset_ * size_of_data_type(dtype_), memory_size(), host_data(),c.events(),c.event("write"));
    }
    void Tensor::to_host(ExecutionContext const &c,bool sync)
    {
        if(cpu_tensor_)
            return;
        c.queue().enqueueReadBuffer(buffer_, sync ? CL_TRUE : CL_FALSE, offset_ * size_of_data_type(dtype_), memory_size(), host_data(),c.events(),c.event("read"));
    }

    Tensor Tensor::sub_tensor(size_t offset,Shape const &s,DataType d,bool trainable) const
    {
        size_t offset_bytes = offset * size_of_data_type(dtype_);
        DLPRIM_CHECK(shape_.total_size()*size_of_data_type(dtype_) >= s.total_size() * size_of_data_type(d));
        DLPRIM_CHECK((offset_ * size_of_data_type(dtype_) + offset_bytes) % size_of_data_type(d) == 0);
        Tensor r;
        r.shape_ = s;
        r.dtype_ = d;
        r.is_trainable_ = trainable;
        r.cpu_tensor_ = cpu_tensor_;
        r.host_ = host_;
        r.buffer_ = buffer_;
        r.capacity_ = r.memory_size();
        r.full_capacity_ = full_capacity_;
        r.cpu_tensor_ = cpu_tensor_;
        r.offset_ = (offset_ * size_of_data_type(dtype_)  + offset_bytes) / size_of_data_type(d);
        return r;
    }

    void *Tensor::host_data()
    {
		if(!host_->p) {
			host_->alloc(full_capacity_);
		}
        return static_cast<char*>(host_->p) + offset_ * size_of_data_type(dtype_);
    }
};
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
