/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#include <dlprim/context.hpp>
#include <sstream>
#include <iostream>

namespace dlprim {
    Context::Context(std::string const &dev_id)
    {
        if(dev_id == "cpu") {
            type_ = cpu;
            return;
        }
        std::istringstream ss(dev_id);
        int p=-1,d=-1;
        char demim = 0;
        ss >>p >> demim >> d;
        if(!ss || demim != ':' || !ss.eof()) {
            throw ValidatioError("Invalid device identification expecting one of `cpu` or `paltform_no:device_no`");
        }
        type_ = gpu;
        select_opencl_device(p,d);
    }

    Context::Context(DeviceType dt,int platform,int device) :
        type_(dt)
    {
        if(dt == cpu)
            return;
        select_opencl_device(platform,device);
    }

    std::string Context::name() const
    {
        if(is_cpu_context())
            return "CPU";
        std::string plat = platform_.getInfo<CL_PLATFORM_NAME>();
        std::string dev  = device_.getInfo<CL_DEVICE_NAME>();
        return dev + " on " + plat;
    }

    void Context::select_opencl_device(int p,int d)
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if(p < 0 || size_t(p) >= platforms.size()) {
            throw ValidatioError("No such platform id " + std::to_string(p) + " total " + std::to_string(platforms.size()) + " avaliblie");
        }
        std::vector<cl::Device> devices;
        platform_ = platforms[p];
        platform_.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if(d < 0 || size_t(d) >= devices.size()) {
            throw ValidatioError("No such device id " + std::to_string(d) + " for platform " 
                                 + std::to_string(p) + " total " + std::to_string(devices.size()) + " avaliblie");
        }
        device_ = devices[d];
        context_ = cl::Context(device_);
    }
    Context::Context(cl::Context const &c,cl::Platform const &p,cl::Device const &d) : 
        platform_(p),
        device_(d),
        context_(c),
        type_(Context::gpu)
    {
    }
}

