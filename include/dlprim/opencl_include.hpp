#pragma once
#include <dlprim/config.hpp>
#ifndef DLPRIM_USE_CL1_HPP
#  ifndef CL_HPP_ENABLE_EXCEPTIONS
#     define CL_HPP_ENABLE_EXCEPTIONS
#  endif
#  ifndef CL_HPP_MINIMUM_OPENCL_VERSION
#    define CL_HPP_MINIMUM_OPENCL_VERSION 120
#  endif
#  ifndef CL_TARGET_OPENCL_VERSION
#    define CL_TARGET_OPENCL_VERSION 120
#  endif
#  ifndef CL_HPP_TARGET_OPENCL_VERSION
#    define CL_HPP_TARGET_OPENCL_VERSION 120
#  endif
#  include <CL/cl2.hpp>
#else
#  ifndef CL_TARGET_OPENCL_VERSION
#    define CL_TARGET_OPENCL_VERSION 120
#  endif
#  ifndef __CL_ENABLE_EXCEPTIONS
#     define __CL_ENABLE_EXCEPTIONS
#  endif
#  include <CL/cl.hpp>
#endif

