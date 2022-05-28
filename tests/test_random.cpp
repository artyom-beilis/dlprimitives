///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/random.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/ops/initialization.hpp>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <limits>
#include "test.hpp"
namespace dp = dlprim;

void test_values(dp::Tensor &t,float mean,float std,float minv,float maxv)
{
    float *p=t.data<float>();
    size_t n=t.shape().total_size();
    double s=0,s2=0;
    for(size_t i=0;i<n;i++) {
        s+=p[i];
        s2+=p[i]*p[i];
        TEST(p[i] >= minv);
        TEST(p[i] <= maxv);
        
    }
    double calc_mean = s/n;
    double calc_std  = std::sqrt(s2/n - calc_mean*calc_mean);
    std::cout << "  mean=" << calc_mean << " std=" << calc_std << std::endl;
    TEST(std::fabs(calc_mean - mean) < 1e-2);
    TEST(std::fabs(calc_std - std) < 1e-2);
}


int main(int argc,char **argv)
{
    try {
        dp::Context ctx(argv[1]);
        dp::ExecutionContext e=ctx.make_execution_context();
        dp::Tensor t(ctx,dp::Shape(10000));
        dp::RandomState st(0xDEADBEEF);
        std::cout << "Testing zero" << std::endl;
        dp::set_to_zero(t,e);
        t.to_host(e);
        test_values(t,0,0.0,0,0);
        std::cout << "Testing constant" << std::endl;
        dp::set_to_constant(t,1.5,e);
        t.to_host(e);
        test_values(t,1.5,0.0,1.5,1.5);
        std::cout << "Testing uniform" << std::endl;
        dp::set_to_urandom(t,st,0.5,2.5,e);
        t.to_host(e);
        test_values(t,1.5,0.5773502691896257,0.5,2.5);
        std::cout << "Testing normal" << std::endl;
        dp::set_to_normal(t,st,-1.5,0.5,e);
        t.to_host(e);
        test_values(t,-1.5,0.5,-std::numeric_limits<float>::max(),std::numeric_limits<float>::max());
        std::cout << "Testing bernoulli" << std::endl;
        dp::set_to_bernoulli(t,st,0.75,e);
        t.to_host(e);
        test_values(t,0.75,std::sqrt(0.75*0.25),0,1);
    }
    catch(std::exception const &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
