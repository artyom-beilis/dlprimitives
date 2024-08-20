///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#include <dlprim/context.hpp>
#include <sstream>
#include <iostream>

namespace dlprim {
    
    Context::Context(ExecutionContext const &ec)
    {
        if(!ec.queue_) {
            type_ = cpu;
            return;
        }
        cl::Context ctx = ec.queue_->getInfo<CL_QUEUE_CONTEXT>();
        cl::Device  dev = ec.queue_->getInfo<CL_QUEUE_DEVICE>();
        // there is no clRetainPlatform this you can construct cl::Platform without true/false flag
        // in fact cl.hpp does not have true/false parameter but cl2.hpp has - but 
        // it is stub that does nothing
        cl::Platform plat(dev.getInfo<CL_DEVICE_PLATFORM>());

        platform_ = plat;
        device_ = dev;
        context_ = ctx;
        type_ = Context::ocl;
    }

    Context::Context(cl::Context const &c,cl::Platform const &p,cl::Device const &d) : 
        platform_(p),
        device_(d),
        context_(c),
        type_(Context::ocl)
    {
    }
    
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
            throw ValidationError("Invalid device identification expecting one of `cpu` or `paltform_no:device_no`");
        }
        type_ = ocl;
        select_opencl_device(p,d);
    }

    Context::Context(ContextType dt,int platform,int device) :
        type_(dt)
    {
        if(dt == cpu)
            return;
        select_opencl_device(platform,device);
    }

    bool Context::check_device_extension(std::string const &name)
    {
        bool res;
        auto p = ext_cache_.find(name);
        if(p == ext_cache_.end()) {
            res = device_extensions().find(name) != std::string::npos;
            ext_cache_[name] = res;
        }
        else {
            res = p->second;
        }
        return res;
    }

    bool Context::is_amd()
    {
        if(is_cpu_context())
            return false;
        return device().getInfo<CL_DEVICE_VENDOR_ID>() == 0x1002;
        //return device_extensions().find("cl_amd_") != std::string::npos;
    }
    bool Context::is_apple()
    {
        if(is_cpu_context())
            return false;
        return device().getInfo<CL_DEVICE_VENDOR_ID>() == 0x1027f00;
    }
    bool Context::is_intel()
    {
        if(is_cpu_context())
            return false;
        return device().getInfo<CL_DEVICE_VENDOR_ID>() == 0x8086;
        //return device_extensions().find("cl_intel_") != std::string::npos;
    }
    bool Context::is_nvidia()
    {
        if(is_cpu_context())
            return false;
        return device().getInfo<CL_DEVICE_VENDOR_ID>() == 0x10DE;
        //return device_extensions().find("cl_nv_") != std::string::npos;
    }
    bool Context::is_imagination()
    {
        if(is_cpu_context())
            return false;
        return device().getInfo<CL_DEVICE_VENDOR_ID>() == 0x1010;
    }

    int Context::estimated_core_count()
    {
        int cu = device().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        if(is_apple())
            // TODO: detect Apple8+ generation, then return cu * 512
            return cu * 256;
        if(is_nvidia())
            return cu * 128;
        if(is_amd())
            return cu * 64;
        if(is_intel())
            return cu * 8;
        return cu;
    }

    std::string const &Context::device_extensions()
    {
        if(is_cpu_context())
            return ext_;
        if(ext_.empty())
            ext_ = device().getInfo<CL_DEVICE_EXTENSIONS>();
        return ext_;
    }


    std::string Context::name() const
    {
        if(is_cpu_context())
            return "CPU";
        std::string plat = platform_.getInfo<CL_PLATFORM_NAME>().c_str();
        std::string dev  = device_.getInfo<CL_DEVICE_NAME>().c_str();
        return dev + " on " + plat;
    }

    void Context::select_opencl_device(int p,int d)
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if(p < 0 || size_t(p) >= platforms.size()) {
            throw ValidationError("No such platform id " + std::to_string(p) + " total " + std::to_string(platforms.size()) + " avaliblie");
        }
        std::vector<cl::Device> devices;
        platform_ = platforms[p];
        platform_.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if(d < 0 || size_t(d) >= devices.size()) {
            throw ValidationError("No such device id " + std::to_string(d) + " for platform " 
                                 + std::to_string(p) + " total " + std::to_string(devices.size()) + " avaliblie");
        }
        device_ = devices[d];
        context_ = cl::Context(device_);
    }
}

