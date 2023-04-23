///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/opencl_include.hpp>
#include <dlprim/definitions.hpp>
#include <chrono>
#include <stack>
#include <map>
#include <memory>

namespace dlprim {

///
/// Class used for benchmarking of the model
///
/// Detailed description TBD
///
class TimingData {
public:
    bool cpu_only=false;
    typedef std::chrono::high_resolution_clock clock_type;
    typedef std::chrono::time_point<clock_type> time_point_type;

    struct Section {
        Section(char const *n) : name(n) {}
        Section(char const *n,time_point_type t) : name(n), start(t) {}
        char const *name="unknown";
        time_point_type start;
        double time_sec;
        int parent = -1;
    };

    struct Data {
        cl::Event event;
        char const *name = nullptr;
        int index = -1;
        int section = -1;
    };

    void enter(char const *name)
    {
        sections_.push_back(Section(name,std::chrono::high_resolution_clock::now()));
        if(!sids_.empty())
            sections_.back().parent = sids_.top();
        sids_.push(sections_.size() - 1);
    }
    void leave()
    {
        int sid = sids_.top();
        auto now = clock_type::now();
        auto diff = now - sections_[sid].start;
        sections_[sid].time_sec = std::chrono::duration_cast<std::chrono::duration<double> >(diff).count();
        sids_.pop();
    }

    void reset()
    {
        sections_.clear();
        while(!sids_.empty())
            sids_.pop();
        events_.clear();
    }

    std::shared_ptr<Data> add_event(char const *name,int index=-1,cl::Event *ev = nullptr)
    {
        std::shared_ptr<Data> e(new Data());
        if(ev)
            e->event = *ev;
        if(!sids_.empty())
            e->section = sids_.top();

        e->name = name;
        e->index = index;
        events_.push_back(e);
        return e;
    }

    std::vector<Section> &sections() {
        return sections_;
    }
    std::vector<std::shared_ptr<Data> > &events() {
        return events_;
    }

private:
    std::vector<Section> sections_;
    std::stack<int> sids_;
    std::vector<std::shared_ptr<Data> > events_;
};


class Context;

///
/// This class is used to pass cl::Events that the kernel should wait for and/or signal event completion
///
/// It is also used to pass cl::CommandQueue over API
///
/// Use it as following:
///
/// \code
/// void do_stuff(...,ExecutionContext const &e)
/// {
///  ...
//// e.queue().enqueueNDRangeKernel(kernel,nd1,nd2,nd3,
///                                   e.events(), // <- events to wait, nullptr of none
///                                   e.event("do_stuff")); //<- event to signal,
///                                                         // if profiling is used will be recorderd
///                                                         // under this name
/// }
/// \endcode
///
///
/// If you need to run several kernels use generate_series_context(#,total)
///
/// For example:
///
/// \code
///   run_data_preparation(e.generate_series_context(0,3)); // waits for events if needed
///   run_processing(e.generate_series_context(1,3)); // no events waited,signaled
///   run_reduce(e.generate_series_context(2,3)); // signals completion event if needed
/// \endcode
class ExecutionContext {
public:
    /// default constructor - can be used for CPU context
    ExecutionContext() :
        event_(nullptr), events_(nullptr) {}

    ///
    /// Create context from cl::CommandQueue, note no events will be waited/signaled
    ///
    ExecutionContext(cl::CommandQueue const &q) :
        queue_(new cl::CommandQueue(q)),event_(nullptr),events_(nullptr)
    {
    }
    ///
    /// Create a context with a request to signal completion event
    ///
    ExecutionContext(cl::CommandQueue const &q,cl::Event *event) :
        queue_(new cl::CommandQueue(q)),event_(event),events_(nullptr)
    {
    }

    ///
    /// Create a context with a request to wait for events
    ///
    ExecutionContext(cl::CommandQueue const &q,std::vector<cl::Event> *events) :
        queue_(new cl::CommandQueue(q)),event_(nullptr),events_(events)
    {
    }
    ///
    /// Create a context with a request to signal completion event and wait for events
    ///
    ExecutionContext(cl::CommandQueue const &q,std::vector<cl::Event> *events,cl::Event *event) :
        queue_(new cl::CommandQueue(q)),event_(event),events_(events)
    {
    }

    bool is_cpu_context() const
    {
        return !queue_;
    }

    ExecutionContext(ExecutionContext const &) = default;
    ExecutionContext &operator=(ExecutionContext const &) = default;

    bool timing_enabled() const
    {
        return !!timing_;
    }

    ///
    /// Add benchmarking/traceing object data
    ///
    void enable_timing(std::shared_ptr<TimingData> p)
    {
        timing_ = p;
    }

    ///
    /// Create contexts for multiple enqueues. 
    ///
    /// The idea is simple if we have events to signal and wait for and multiple
    /// kernels to execute, the first execution id == 0 should provide list of events
    /// to wait if id == total - 1, give event to signal
    ///
    ExecutionContext generate_series_context(size_t id,size_t total) const
    {
        ExecutionContext ctx = generate_series_context_impl(id,total);
        ctx.timing_ = timing_;
        return ctx;
    }

    ///
    /// Profiling scope enter called by ExecGuard::ExecGuard()
    ///
    void enter(char const *name) const
    {
        if(timing_)
            timing_->enter(name);
    }
    ///
    /// Profiling scope leave, called by ExecGuard::~ExecGuard()
    ///
    void leave() const
    {
        if(timing_)
            timing_->leave();
    }

    void finish()
    {
        if(queue_)
            queue_->finish();
    }

    
    ///
    /// Get the command queue. Never call it in non-OpenCL context
    ///
    cl::CommandQueue &queue() const
    {
        DLPRIM_CHECK(queue_);
        return *queue_;
    }

    ///
    /// Get event to signal. Note: name is used for profiling. Such that profiling
    /// is enabled profiling conters will be written the TimingData with the name of the
    /// kernel you call. Optional id allows to distinguish between multiple similar calls
    ///
    cl::Event *event(char const *name = "unknown", int id = -1) const
    {
        if(timing_ && !timing_->cpu_only) {
            return &timing_->add_event(name,id,event_)->event;
        }
        return event_;
    }
    ///
    /// Get events to wait for
    ///
    std::vector<cl::Event> *events() const {
        return events_;
    }


    ///
    /// Create context that waits for event if needed - use only if you know that more kernels are followed
    ///
    ExecutionContext first_context() const
    {
        if(queue_ == nullptr)
            return ExecutionContext();
        return ExecutionContext(queue(),events_);
    }

    ///
    /// Create context does not wait or signals use only if you know that more kernels run before and after 
    ///
    ExecutionContext middle_context() const
    {
        if(queue_ == nullptr)
            return ExecutionContext();
        return ExecutionContext(queue());
    }
    ///
    /// Create context that signals for completion event if needed - use only if you know that more kernels run before
    ///
    ExecutionContext last_context() const
    {
        if(queue_ == nullptr)
            return ExecutionContext();
        return ExecutionContext(queue(),event_);
    }

private:
    ExecutionContext generate_series_context_impl(size_t id,size_t total) const
    {
        if(total <= 1)
            return *this;
        if(id == 0)
            return first_context();
        if(id + 1 >= total)
            return last_context();
        return middle_context();
    }


    std::shared_ptr<TimingData> timing_;
    std::shared_ptr<cl::CommandQueue> queue_; /// make sure copying is fast
    cl::Event *event_;
    std::vector<cl::Event> *events_;
    friend class Context;
};


///
/// This is main object that represent the pair of OpenCL platform and device
/// all other objects use it.
///
/// It can be CPU context - meaning that it represents no OpenCL platform/device/context
///
///
class Context {
public:
    /// Device used with the context, CPU or OpenCL device.
    enum ContextType {
        cpu = 0, ///< CPU no OpenCL platform/device/context are used
        ocl = 1, ///< Use OpenCL device, it also may be CPU device.
    };


    ///
    /// Create new context from textual ID. It can be "cpu" or "P:D"
    /// were P is integer representing platform and D is device number on this platform
    /// starting from 0, for example "0:1" is second device on 1st platform.
    ///
    Context(std::string const &dev_id);
    ///
    /// Create context from numerical platform and device number, of it is CPU context,
    /// platform and device are ignored
    ///
    Context(ContextType dt = cpu,int platform = 0,int device = 0);
    ///
    /// Create the object from OpenCL context, platform and device..
    ///
    Context(cl::Context const &c,cl::Platform const &p,cl::Device const &d);

    /// 
    /// Create the object from queue
    ///
    Context(ExecutionContext const &ec);

    Context(Context const &) = default;
    Context &operator=(Context const &) = default;
    Context(Context &&) = default;
    Context &operator=(Context &&) = default;
    ~Context() {}

    ///
    /// Human readable name for the context, for example:
    /// "GeForce GTX 960 on NVIDIA CUDA"
    ///
    std::string name() const;

    /// return context type either cpu or ocl
    ContextType context_type() const;

    /// Returns true if the context was created as CPU context 
    bool is_cpu_context() const
    {
        return type_ == cpu;
    }
    /// Returns true if the context was created as OpenCL context
    bool is_opencl_context() const
    {
        return type_ == ocl;
    }
    /// Get OpenCL platform object
    cl::Platform &platform()
    {
        return platform_;
    }
    /// Get OpenCL device object
    cl::Device &device()
    {
        return device_;
    }

    /// Check if specific device extension is present
    bool check_device_extension(std::string const &name);
    
    /// get all device extensions as a string
    std::string const &device_extensions();

    ///
    /// Get estimated number of cores. Note since it is not something defined for OpenCL in general
    /// it returns number of cuda cores for NVidia devices and similar values for AMD and Intel GPU
    /// devices. For Nvidia it is 128 * cu, for AMD it is 64 * cu and for Intel it is 8 * cu where
    /// cu is number of compute units reported by `CL_DEVICE_MAX_COMPUTE_UNITS` query 
    /// 
    int estimated_core_count();

    /// checks if the device is AMD GPU
    bool is_amd();
    /// checks if the device is Apple GPU
    bool is_apple();
    /// checks if the device is NVidia GPU
    bool is_nvidia();
    /// checks if the device is Intel GPU
    bool is_intel();
    /// checks if the device is Imagination GPU
    bool is_imagination();

    /// Get OpenCL context object
    cl::Context &context()
    {
        return context_;
    }
    /// Creates a new Command queue for the context with optional properties
    cl::CommandQueue make_queue(cl_command_queue_properties props=0)
    {
        cl::CommandQueue q;
        if(!is_cpu_context())
            q=std::move(cl::CommandQueue(context_,device_,props));
        return q;
    }

    /// Generate ExecutionContext (queue + events)
    ExecutionContext make_execution_context(cl_command_queue_properties props=0)
    {
        if(is_cpu_context())
            return ExecutionContext();
        else
            return ExecutionContext(make_queue(props));
    }

private:
    void select_opencl_device(int p,int d);
    cl::Platform platform_;
    cl::Device device_;
    cl::Context context_;
    ContextType type_;;
    std::map<std::string,bool> ext_cache_;
    std::string ext_;
};



class ExecGuard {
public:
    ExecGuard(ExecGuard const &) = delete;
    void operator=(ExecGuard const &) = delete;
    ExecGuard(ExecutionContext const &ctx,char const *name) : ctx_(&ctx)
    {
        ctx_->enter(name);
    }
    ~ExecGuard()
    {
        ctx_->leave();
    }
private:
    ExecutionContext const *ctx_;
};


} // namespace
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

