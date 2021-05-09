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
    private:
        void select_opencl_device(int p,int d);
        cl::Platform platform_;
        cl::Device device_;
        cl::Context context_;
        DeviceType type_;;
        
    };


} // namespace
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

