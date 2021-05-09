#pragma once
#include <dlprim/context.hpp>
#include <mutex>
#include <map>
#include <unordered_map>

namespace dlprim {
    namespace gpu {

        extern std::map<std::string,std::string> kernel_sources;

        struct Parameter {
            Parameter(std::string const &n,int v):
                name(n), value(std::to_string(v))
            {
            }
            Parameter(std::string const &n,std::string const &v):
                name(n), value(v)
            {
            }

            std::string name;
            std::string value;
        };

        class Cache {
        public:
            static Cache &instance();
            cl::Program const &get_program(Context  &ctx,std::string const &source,std::vector<Parameter> const &params);
            static cl::Program build_program(Context &ctx,std::string const &source,std::vector<Parameter> const &params);
        private:
            static std::string make_key(Context &ctx,std::string const &src,std::vector<Parameter> const &params);
            std::unordered_map<std::string,cl::Program> cache_;
            std::mutex mutex_;
        };
    }
}
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

