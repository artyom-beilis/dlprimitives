#include  <dlprim/gpu/program_cache.hpp>
#include <sstream>

namespace dlprim {
namespace gpu {

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
    cl::Program prg(ctx.context(),ks->second);
    std::ostringstream ss;
    std::string ocl_version = ctx.platform().getInfo<CL_PLATFORM_VERSION>();
    if(ocl_version.substr(7,1) >= "2") 
	    ss << "-cl-std=CL2.0 ";
    for(size_t i=0;i<params.size();i++) {
        if(i > 0)
            ss<<" ";
        ss << "-D" << params[i].name <<"=" <<params[i].value;
    }
    try {
        prg.build(std::vector<cl::Device>{ctx.device()},ss.str().c_str());
    }
    catch(cl::BuildError const &e) {
        std::string log;
        auto cl_log = e.getBuildLog();
        for(size_t i=0;i<cl_log.size();i++) {
            log += "For device: ";
            log += cl_log[i].first.getInfo<CL_DEVICE_NAME>();
            log += "\n";
            log += cl_log[i].second;
        }
        throw BuildError("Failed to build program source " + source + " with parameters " + ss.str() + " log:\n" + log.substr(0,1024),log);
    }
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
