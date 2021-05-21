#pragma once
#include <dlprim/opencl_include.hpp>
#include <dlprim/definitions.hpp>

namespace dlprim {

    class Context {
    public:
        enum DeviceType {
            cpu = 0,
            gpu = 1,
        };
        

        Context(std::string const &dev_id);
        Context(DeviceType dt = cpu,int platform = 0,int device = 0);
        Context(cl::Context const &c,cl::Platform const &p,cl::Device const &d);

        Context(Context const &) = default;
        Context &operator=(Context const &) = default;
        Context(Context &&) = default;
        Context &operator=(Context &&) = default;
        ~Context() {}

        std::string name() const;

        DeviceType device_type() const;

        bool is_cpu_context() const
        {
            return type_ == cpu;
        }
        bool is_gpu_context() const
        {
            return type_ == gpu;
        }
        cl::Platform &platform()
        {
            return platform_;
        }
        cl::Device &device()
        {
            return device_;
        }

        cl::Context &context()
        {
            return context_;
        }
        cl::CommandQueue make_queue()
        {
            cl::CommandQueue q;
            if(!is_cpu_context())
                q=std::move(cl::CommandQueue(context_,device_));
            return q;
        }
    private:
        void select_opencl_device(int p,int d);
        cl::Platform platform_;
        cl::Device device_;
        cl::Context context_;
        DeviceType type_;;
        
    };

    class ExecutionContext {
    public:
        ExecutionContext() :
            queue_(nullptr),event_(nullptr), events_(nullptr) {}
        ExecutionContext(cl::CommandQueue &q) : 
            queue_(&q),event_(nullptr),events_(nullptr)
        {
        }
        ExecutionContext(cl::CommandQueue &q,cl::Event *event) : 
            queue_(&q),event_(event),events_(nullptr)
        {
        }
        ExecutionContext(cl::CommandQueue &q,std::vector<cl::Event> *events) : 
            queue_(&q),event_(nullptr),events_(events)
        {
        }
        ExecutionContext(cl::CommandQueue &q,std::vector<cl::Event> *events,cl::Event *event) : 
            queue_(&q),event_(event),events_(events)
        {
        }

        ExecutionContext generate_series_context(size_t id,size_t total) const
        {
            if(total <= 1)
                return *this;
            if(id == 0)
                return first_context();
            if(id + 1 >= total)
                return last_context();
            return middle_context();
        }

        ExecutionContext first_context() const
        {
            if(queue_ == nullptr)
                return ExecutionContext();
            return ExecutionContext(queue(),events_);
        }
        
        ExecutionContext middle_context() const
        {
            if(queue_ == nullptr) 
                return ExecutionContext();
            return ExecutionContext(queue());
        }
         
        ExecutionContext last_context() const
        {
            if(queue_ == nullptr)
                return ExecutionContext();
            return ExecutionContext(queue(),event_);
        }

        cl::CommandQueue &queue() const
        { 
            DLPRIM_CHECK(queue_ != nullptr);
            return *queue_; 
        }
        cl::Event *event() const { return nullptr; }
        std::vector<cl::Event> *events() const { return nullptr; }
    private:
        cl::CommandQueue *queue_;
        cl::Event *event_;
        std::vector<cl::Event> *events_;
    };


} // namespace
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

