///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include  <dlprim/gpu/program_cache.hpp>
#include <sstream>
#include <chrono>
#include <iostream>
#ifdef WITH_SQLITE3
#include "binary_cache.hpp"
#endif


//#define DEBUG_CACHE_TIMES 

namespace dlprim {
namespace gpu {


#ifdef DEBUG_CACHE_TIMES
class TimeWriter {
public:
    decltype(std::chrono::high_resolution_clock::now()) start;
    std::string name;
    TimeWriter(std::string const &n) : name(n)
    {
        start = std::chrono::high_resolution_clock::now();
    }
    ~TimeWriter()
    {
        auto stop = std::chrono::high_resolution_clock::now();
        auto passed = std::chrono::duration_cast<std::chrono::duration<double> > ((stop-start)).count();
        std::cout << "Kernel " << name << " " << passed * 1e3 << " ms" << std::endl;
    }
};
#endif

Cache &Cache::instance()
{
    static Cache c;
    return c;
}


cl::Program const &Cache::get_program(Context &ctx,std::string const &source,std::vector<Parameter> const &params)
{
    std::string key = make_key(ctx,source,params);
    std::unique_lock<std::mutex> g(mutex_);
    auto p = cache_.find(key);
    if(p == cache_.end()) {
        auto prg = build_program(ctx,source,params);
        cache_[key]=prg;
    }
    return cache_[key];

}


cl::Program Cache::build_program(Context  &ctx,std::string const &source,std::vector<Parameter> const &params)
{
    auto ks = kernel_sources.find(source);
    if(ks == kernel_sources.end())
        throw ValidationError("Unknow program source " + source);
    std::string const &source_text = ks->second;
    std::ostringstream prepend;
    std::ostringstream ss;
    bool combine = false;
    std::string ocl_version = ctx.platform().getInfo<CL_PLATFORM_VERSION>();
    if(ocl_version.substr(7,1) >= "2") 
	    ss << "-cl-std=CL2.0 ";
    bool enable_half = ctx.check_device_extension("cl_khr_fp16");
    for(size_t i=0;i<params.size();i++) {
        if(i > 0)
            ss<<" ";
        if(params[i].name.c_str()[0]=='#') {
            prepend << "#define " << params[i].name.c_str() + 1 << " " << params[i].value << "\n";
            combine=true;
        }
        else {
            ss << "-D" << params[i].name <<"=" <<params[i].value;
        }
    }
    if(enable_half) {
        prepend << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        combine=true;
    }
    std::string const &code = (combine ? prepend.str() + source_text : source_text);
    std::string sparams = ss.str();
    
    #ifdef DEBUG_CACHE_TIMES
    TimeWriter guard(source);
    #endif

    #ifdef  WITH_SQLITE3
    /// nvidia has very different type of caching since binary return not binary but rather ptx,
    /// using only native cuda cache works better, double cache adds overhead
    bool use_cache = !ctx.is_nvidia();
    if(use_cache) {
        auto &pc = BinaryProgramCache::instance();
        auto binary = pc.get_binary(ctx,code,sparams);
        if(!binary.empty()) {
            cl::Program::Binaries bin(1);
            bin[0].swap(binary);
            try {
                cl::Program prg(ctx.context(),{ctx.device()},bin);
                prg.build();
                return prg;
            }
            catch(cl::Error const &e) {
                throw BuildError("Failed to build program binary" + std::string(e.what()),source);
            }
        }
    }
    #endif
    cl::Program prg(ctx.context(),code);
    try {
        prg.build(std::vector<cl::Device>{ctx.device()},sparams.c_str());
    }
    #ifndef DLPRIM_USE_CL1_HPP
    catch(cl::BuildError const &e) {
        std::string log;
        auto cl_log = e.getBuildLog();
        for(size_t i=0;i<cl_log.size();i++) {
            log += "For device: ";
            log += cl_log[i].first.getInfo<CL_DEVICE_NAME>();
            log += "\n";
            log += cl_log[i].second;
        }
        std::cerr <<"Failed Program Code:\n"
                    "=========================\n"
                    << code << 
                    "=========================\n" <<std::endl;

        throw BuildError("Failed to build program source " + source + " with parameters " + ss.str() + " log:\n"
                    + log.substr(0,1024) + "\nclBuildErrorCode: " + std::to_string(e.err()),log);
    }
    #else
    catch(cl::Error const &e) {
        char buffer[1024];
        size_t len=0;
        clGetProgramBuildInfo(prg(), ctx.device()(), CL_PROGRAM_BUILD_LOG, sizeof(buffer)-1, buffer, &len);
        buffer[std::min(len,sizeof(buffer)-1)]=0;
        throw BuildError("Failed to build program source " + source + " with parameters " + ss.str() + " log:\n" + buffer,buffer);
    }
    #endif
    #ifdef  WITH_SQLITE3
    if(use_cache){
        auto binaries = prg.getInfo<CL_PROGRAM_BINARIES>();
        DLPRIM_CHECK(binaries.size() == 1);
        auto &pc = BinaryProgramCache::instance();
        pc.save_binary(ctx,code,sparams,binaries[0],source);
    }
    #endif

    return prg;
}


std::string Cache::make_key(Context &ctx,std::string const &src,std::vector<Parameter> const &params)
{
    void *ctx_ptr = ctx.context()();
    std::ostringstream ss;
    ss << "prg:" << ctx_ptr <<  "@" << src <<  "/?";
    for(size_t i=0;i<params.size();i++) {
        if(i > 0)
            ss << '&';
        ss << params[i].name << '=' << params[i].value;
    }
    return ss.str();
}


}
}
