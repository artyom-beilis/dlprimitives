///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/config.hpp>
#if !defined(DLPRIM_USE_CL1_HPP) && !defined(DLPRIM_USE_CL2_HPP)
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
#  ifdef __APPLE__
#    include <OpenCL/opencl.hpp>
#  else
#    include <CL/opencl.hpp>
#  endif
#elif defined(DLPRIM_USE_CL2_HPP)
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
#  ifdef __APPLE__
#    include <OpenCL/cl2.hpp>
#  else
#    include <CL/cl2.hpp>
#  endif
#else
#  ifndef CL_TARGET_OPENCL_VERSION
#    define CL_TARGET_OPENCL_VERSION 120
#  endif
#  ifndef __CL_ENABLE_EXCEPTIONS
#     define __CL_ENABLE_EXCEPTIONS
#  endif
#  ifdef __APPLE__
#    include <OpenCL/cl.hpp>
#  else
#    include <CL/cl.hpp>
#  endif
#endif

