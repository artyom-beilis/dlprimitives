///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <array>
#include <dlprim/definitions.hpp>
#include <cmath>

namespace dlprim {
    /// CPU related tools and classes
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
                case StandardActivations::tanh:
                    {
                        for(size_t i=0;i<n;i++) {
                            p[i] = std::tanh(p[i]);
                        }
                    }
                    break;
                case  StandardActivations::sigmoid:
                    {
                        for(size_t i=0;i<n;i++) {
                            p[i] = 1 / (1 + std::exp(-p[i]));
                        }
                    }
                    break;
                case StandardActivations::relu6:
                    {
                        T zero=T();
                        for(size_t i=0;i<n;i++) {
                            p[i] = std::min(std::max(p[i],zero),T(6));
                        }
                    }
                    break;
            };
        }

        template<typename T>
        inline void apply_activation_diff(size_t n,T const *y,T const *dy,T *dx,StandardActivations a)
        {
            switch(a) {
                case StandardActivations::identity:
                    {
                        for(size_t i=0;i<n;i++) {
                            dx[i] = dy[i];
                        }
                    }
                    break;
                case StandardActivations::relu:
                    {
                        for(size_t i=0;i<n;i++) {
                            dx[i] = y[i] > 0 ? dy[i] : 0;
                        }
                    }
                    break;
                case StandardActivations::tanh:
                    {
                        for(size_t i=0;i<n;i++) {
                            T yv = y[i];
                            dx[i] = dy[i] * (1-yv*yv);
                        }
                    }
                    break;
                case  StandardActivations::sigmoid:
                    {
                        for(size_t i=0;i<n;i++) {
                            T yv = y[i];
                            dx[i] = dy[i] * yv * (1-yv);
                        }
                    }
                    break;
                case StandardActivations::relu6:
                    {
                        for(size_t i=0;i<n;i++) {
                            dx[i] = (6 > y[i] && y[i] > 0) ? dy[i] : 0;
                        }
                    }
                    break;
            };
        }

        template<typename T>
        inline void apply_activation_diff(size_t n,T const *y,T const *dy,T *dx,float beta,StandardActivations a)
        {
            if(beta == 0.0f) {
                apply_activation_diff(n,y,dy,dx,a);
                return;
            }
            switch(a) {
                case StandardActivations::identity:
                    {
                        for(size_t i=0;i<n;i++) {
                            dx[i] = beta*dx[i] + dy[i];
                        }
                    }
                    break;
                case StandardActivations::relu:
                    {
                        for(size_t i=0;i<n;i++) {
                            dx[i] = beta*dx[i] + (y[i] > 0 ? dy[i] : 0);
                        }
                    }
                    break;
                case StandardActivations::tanh:
                    {
                        for(size_t i=0;i<n;i++) {
                            T yv = y[i];
                            dx[i] = beta*dx[i] + (dy[i] * (1-yv*yv));
                        }
                    }
                    break;
                case  StandardActivations::sigmoid:
                    {
                        for(size_t i=0;i<n;i++) {
                            T yv = y[i];
                            dx[i] = beta*dx[i] + (dy[i] * yv * (1-yv));
                        }
                    }
                    break;
                case StandardActivations::relu6:
                    {
                        for(size_t i=0;i<n;i++) {
                            dx[i] = beta*dx[i] + (6 > y[i] && y[i] > 0 ? dy[i] : 0);
                        }
                    }
                    break;
            };
        }
    }
};
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
