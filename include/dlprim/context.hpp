#pragma once
#include <dlprim/opencl_include.hpp>
#include <dlprim/definitions.hpp>
#include <chrono>
#include <stack>
#include <map>
#include <memory>

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

        bool check_device_extension(std::string const &name);
        std::string const &device_extensions();

        cl::Context &context()
        {
            return context_;
        }
        cl::CommandQueue make_queue(cl_command_queue_properties props=0)
        {
            cl::CommandQueue q;
            if(!is_cpu_context())
                q=std::move(cl::CommandQueue(context_,device_,props));
            return q;
        }
    private:
        void select_opencl_device(int p,int d);
        cl::Platform platform_;
        cl::Device device_;
        cl::Context context_;
        DeviceType type_;;
        std::map<std::string,bool> ext_cache_;
        std::string ext_;
    };

    class TimingData {
    public:
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
        
        std::vector<Section> &sections() { return sections_; }
        std::vector<std::shared_ptr<Data> > &events() { return events_; }

    private:
        std::vector<Section> sections_;
        std::stack<int> sids_;
        std::vector<std::shared_ptr<Data> > events_;
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
        
        bool timing_enabled() const
        {
            return !!timing_;
        }

        void enable_timing(std::shared_ptr<TimingData> p)
        {
            timing_ = p;
        }
        
        ExecutionContext generate_series_context(size_t id,size_t total) const
        {
            ExecutionContext ctx = generate_series_context_impl(id,total);
            ctx.timing_ = timing_;
            return ctx;
        }

        void enter(char const *name) const
        {
            if(timing_)
                timing_->enter(name);
        }
        void leave() const
        {
            if(timing_)
                timing_->leave();
        }

        cl::CommandQueue &queue() const
        { 
            DLPRIM_CHECK(queue_ != nullptr);
            return *queue_; 
        }   
        cl::Event *event(char const *name = "unknown", int id = -1) const 
        { 
            if(timing_) {
                return &timing_->add_event(name,id,event_)->event;
            }
            return event_;
        }
        std::vector<cl::Event> *events() const { return events_; }
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

        std::shared_ptr<TimingData> timing_;
        cl::CommandQueue *queue_;
        cl::Event *event_;
        std::vector<cl::Event> *events_;
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

