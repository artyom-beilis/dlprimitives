#pragma once
#include <array>
#include <dlprim/definitions.hpp>

namespace dlprim {
    namespace cpu {
        template<typename T>
        inline void apply_activation(T *p,size_t n,StandardActivations a)
        {
            switch(a) {
                case StandardActivations::identity: 
                    break;
                case StandardActivations::relu:
                    {
                        T zero=T();
                        for(size_t i=0;i<n;i++) {
                            p[i] = std::max(p[i],zero);
                        }
                    }
                    break;
            };
        }
    }
};
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
