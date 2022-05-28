///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <string>
#include <stdexcept>
#include <sstream>
#include <cmath>

#define TEST(x) do { if(!(x)) throw std::runtime_error(#x " failed in line " + std::to_string(__LINE__)); } while(0)

#define TESTEQ(a,b) do { \
        decltype(a) l=(a), r=(b); \
        if(l!=r) { \
            std::ostringstream ss; \
            ss << "Failed: " #a " == " #b " at line " << __LINE__ << ":" << l << "!=" << r << std::endl; \
            throw  std::runtime_error(ss.str()); \
        }} while(0) 

#define TESTEQF(a,b,eps) do { \
        auto l=(a); auto r=(b); \
        if(std::abs(l-r) > eps) { \
            std::ostringstream ss; \
            ss << "Failed: " #a " ~= " #b " at line " << __LINE__ << ":" << l << "!=" << r << std::endl; \
            throw  std::runtime_error(ss.str()); \
        }} while(0) 
