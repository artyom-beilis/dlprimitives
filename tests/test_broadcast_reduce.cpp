///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/core/pointwise.hpp>
#include <iostream>
#include "test.hpp"

namespace dp = dlprim;

void test_shape()
{
    std::cout << "Test Shape" << std::endl;
    dp::Shape s(2,3,4);
    TEST(s.unsqueeze(0) == dp::Shape(1,2,3,4));
    TEST(s.unsqueeze(3) == dp::Shape(2,3,4,1));
    TEST(s.unsqueeze(1) == dp::Shape(2,1,3,4));
    TEST(s.unsqueeze(-1) == dp::Shape(2,3,4,1));

    TEST(dp::Shape(2,4,1,1).squeeze() == dp::Shape(2,4));
    TEST(dp::Shape(2,4,1,1).squeeze({ 2, 3}) == dp::Shape(2,4));
    TEST(dp::Shape(2,4,1,1).squeeze({-1,-2}) == dp::Shape(2,4));
    TEST(dp::Shape(2,1,4,1).squeeze() == dp::Shape(2,4));
    TEST(dp::Shape(2,1,4,1).squeeze({ 1,-1}) == dp::Shape(2,4));
    TEST(dp::Shape(2,1,4,1).squeeze({ 1, 3}) == dp::Shape(2,4));

    TEST(dp::Shape(2,3,4).reshape({0,-1}) == dp::Shape(2,12));
    TEST(dp::Shape(2,3,4).reshape({-1,12}) == dp::Shape(2,12));
    TEST(dp::Shape(2,3,4).reshape({2,12}) == dp::Shape(2,12));
    TEST(dp::Shape(2,3,4).reshape({-1,4}) == dp::Shape(6,4));

    std::cout << "Broadcasting" << std::endl;
    TEST(dp::broadcast(dp::Shape(2,3,4),dp::Shape(3,4)) == dp::Shape(2,3,4));
    TEST(dp::broadcast(dp::Shape(1,3,4),dp::Shape(5,3,1)) == dp::Shape(5,3,4));
    TEST(dp::broadcast(dp::Shape(1,3,1),dp::Shape(2,3,4)) == dp::Shape(2,3,4));
    
    std::cout << "Strides" << std::endl;

    TEST(dp::Shape(3,4).broadcast_strides(dp::Shape(2,3,4)) == dp::Shape(0,4,1));
    TEST(dp::Shape(3,1,1).broadcast_strides(dp::Shape(8,3,32,32)) == dp::Shape(0,1,0,0));
    TEST(dp::Shape(5,1).broadcast_strides(dp::Shape(5,5)) == dp::Shape(1,0));

    std::cout << "Shrink" << std::endl;
    std::vector<dp::Shape> src,tgt;
    
    src={dp::Shape(2,3,4),dp::Shape(2,1,1)};
    tgt={dp::Shape(2,12),dp::Shape(2,1)};
    dp::shrink_broadcast_ranges(src);
    TEST(src==tgt);

    src={dp::Shape(2,3,4),dp::Shape(2,3,4)};
    tgt={dp::Shape(24),dp::Shape(24)};
    dp::shrink_broadcast_ranges(src);
    TEST(src==tgt);

    src={dp::Shape(2,3,4),dp::Shape(1)};
    tgt={dp::Shape(24),dp::Shape(1)};
    dp::shrink_broadcast_ranges(src);
    TEST(src==tgt);

    src={dp::Shape(2,3,4),dp::Shape(3,1)};
    tgt={dp::Shape(2,3,4),dp::Shape(1,3,1)};
    dp::shrink_broadcast_ranges(src);
    TEST(src==tgt);

    src={dp::Shape(2,3,4,5),dp::Shape(3,1,1)};
    tgt={dp::Shape(2,3,20),dp::Shape(1,3,1)};
    dp::shrink_broadcast_ranges(src);
    TEST(src==tgt);

    src={dp::Shape(2,3,4,5),dp::Shape(1,3,4,1)};
    tgt={dp::Shape(2,12,5),dp::Shape(1,12,1)};
    dp::shrink_broadcast_ranges(src);
    TEST(src==tgt);

    src={dp::Shape(5),dp::Shape(5)};
    tgt={dp::Shape(5),dp::Shape(5)};
    dp::shrink_broadcast_ranges(src);
    TEST(src==tgt);
}

#define HALF_MANTISSA(x) (((x) & 1023) | (((x) & 0x7C00) == 0 ? 0 : 1024))
#define HALF_EXPONENT(x) (((x) & 0x7C00) >> 10)
#define HALF_SIGN_MASK 0x8000
struct my_half {
    uint16_t val;
    my_half(int v = 0)
    {
        val = f16_from_int(v);
    }
    int to_int() const
    {
        short a=val;
        unsigned short value = HALF_MANTISSA(a);
        short shift = HALF_EXPONENT(a) - 25;
        if(shift > 0)
            value <<= shift;
        else if(shift < 0)
            value >>= -shift;
        if(a & HALF_SIGN_MASK)
            return -(int32_t)(value);
        return value;

    }

    my_half &operator+=(int v)
    {
        *this = my_half(to_int() + v);
        return *this;
    }
    uint16_t f16_from_int(int32_t sv)
    {
        uint32_t v;
        int sig = 0;
        if(sv < 0) {
            v=-sv;
            sig=1;
        }
        else
            v=sv;
        if(v==0)
            return 0;
        int e=25;
        while(v >= 2048) {
            v>>=1;
            e++;
        }
        while(v<1024) {
            v<<=1;
            e--;
        }
        if(e>=31)
            return  0x7C00 | (sig << 15);
        return (sig << 15) | (e << 10) | (v & 1023);
    }
    bool operator==(my_half const &other) const
    {
        return val == other.val;
    }
    bool operator!=(my_half const &other) const
    {
        return !(*this == other);
    }
};

namespace dlprim {
    template<>
    struct TypeTraits<my_half> { static constexpr DataType data_type = half_data; };
}


template<typename T>
dp::Tensor make_tensor(dp::ExecutionContext const &q,dp::Shape s,std::vector<T> values)
{
    dp::Context ctx(q);
    TEST(s.total_size() == values.size());
    TEST(values.size() > 0);
    dp::Tensor r(ctx,s,dp::TypeTraits<T>::data_type);
    r.to_device(q,values.data());
    return r;
}


bool equal(dp::Tensor a,dp::Tensor b,dp::ExecutionContext const &q,int eps = 0)
{
    if(a.shape() != b.shape() || a.dtype() != b.dtype()) {
        return false;
    }
    a.to_host(q);
    b.to_host(q);
    bool res;
    if(eps == 0 || a.dtype() != dlprim::half_data) {
        res = memcmp(a.host_data(),b.host_data(),a.memory_size())==0;
    }
    else  {
        res = true;
        size_t N = a.shape().total_size();
        for(size_t i=0;i<N;i++) {
            int a_val = a.data<my_half>()[i].to_int();
            int b_val = b.data<my_half>()[i].to_int();
            if(abs(a_val - b_val) > eps) {
                std::cout << "Failure for half at " << i << " "<<a_val << "!="<<b_val << "withing " << eps << std::endl;
                res = false;
                break;
            }
        }
    }
    if(!res) {
           std::cout << "Failed for tensors " << a << "==" << b << std::endl;
        if(a.dtype() == dlprim::float_data) {
            for(size_t i=0;i<a.shape().total_size();i++) {
                std::cout << i << ": " << a.data<float>()[i] << " " << b.data<float>()[i] << std::endl;
            }
        }
        else if(a.dtype() == dlprim::int64_data) {
            for(size_t i=0;i<a.shape().total_size();i++) {
                std::cout << i << ": " << a.data<int64_t>()[i] << " " << b.data<int64_t>()[i] << std::endl;
            }
        }
        /*else if(a.dtype() == dlprim::half_data) {
            for(size_t i=0;i<a.shape().total_size();i++) {
                std::cout << i << ": " << a.data<my_half>()[i].to_int() << " " << b.data<my_half>()[i].to_int() << std::endl;
            }
        }*/
    }
    return res;
}


template<typename Type>
void test_pointwise(dp::ExecutionContext const &q)
{
    using dp::core::pointwise_operation;
    dp::Context ctx(q);
    {
        auto a=make_tensor<Type>(q,dp::Shape(3,2),{1,2,3,4,5,6});
        auto b=make_tensor<Type>(q,dp::Shape(3,2),{5,6,7,8,9,10});
        auto ref=make_tensor<Type>(q,dp::Shape(3,2),{6,8,10,12,14,16});
        dp::Tensor c(ctx,dp::Shape(3,2),a.dtype());
        std::cout << a <<"+"<<b<<"->"<<c<<std::endl;
        pointwise_operation({a,b},{c},{},"y0=x0+x1;",q);
        TEST(equal(c,ref,q));
    }
    {
        auto a=make_tensor<Type>(q,dp::Shape(3,2),{1,2,3,4,5,6});
        auto b=make_tensor<Type>(q,dp::Shape(3,2),{5,6,7,8,9,10});
        auto ref=make_tensor<Type>(q,dp::Shape(3,2),{-4,-4,-4,-4,-4,-4});
        dp::Tensor c(ctx,dp::Shape(3,2),a.dtype());
        std::cout << a <<"+"<<b<<"->"<<c<<std::endl;
        pointwise_operation({a,b},{c},{-1},"y0=x0+w0*x1;",q);
        TEST(equal(c,ref,q));
    }
}

template<typename Type>
void test_broadcast(dp::ExecutionContext const &q)
{
    using dp::core::pointwise_operation_broadcast;
    dp::Context ctx(q);
    {
        auto a=make_tensor<Type>(q,dp::Shape(1),{1});
        auto b=make_tensor<Type>(q,dp::Shape(2),{2,3});
        auto ref=make_tensor<Type>(q,dp::Shape(2),{3,4});
        dp::Tensor c(ctx,dp::Shape(2),a.dtype());
        std::cout << a <<"+"<<b<<"->"<<c<<std::endl;
        pointwise_operation_broadcast({a,b},{c},{},"y0=x0+x1;",q);
        TEST(equal(c,ref,q));
    }
    {
        auto a=make_tensor<Type>(q,dp::Shape(1,2),{0,1});
        auto b=make_tensor<Type>(q,dp::Shape(2,1),{0,1});
        auto ref=make_tensor<Type>(q,dp::Shape(2,2),{0,1,1,2});
        dp::Tensor c(ctx,dp::Shape(2,2),a.dtype());
        std::cout << a <<"+"<<b<<"->"<<c<<std::endl;
        pointwise_operation_broadcast({a,b},{c},{},"y0=x0+x1;",q);
        TEST(equal(c,ref,q));
    }
    {
        auto a=make_tensor<Type>(q,dp::Shape(1,2),{0,1});
        auto b=make_tensor<Type>(q,dp::Shape(2,1),{0,1});
        auto ref=make_tensor<Type>(q,dp::Shape(2,2),{0,1,7,8});
        dp::Tensor c(ctx,dp::Shape(2,2),a.dtype());
        std::cout << a <<"+"<<b<<"->"<<c<<std::endl;
        pointwise_operation_broadcast({a,b},{c},{7},"y0=x0+w0*x1;",q);
        TEST(equal(c,ref,q));
    }
    {
        auto a=make_tensor<Type>(q,dp::Shape(1,2,1),{0,1});
        auto b=make_tensor<Type>(q,dp::Shape(2,1,3),{0,1,2,3,4,5,});
        auto ref=make_tensor<Type>(q,dp::Shape(2,2,3),{0, 1, 2, 1, 2, 3, 3, 4, 5, 4, 5, 6});
        dp::Tensor c(ctx,dp::Shape(2,2,3),a.dtype());
        std::cout << a <<"+"<<b<<"->"<<c<<std::endl;
        pointwise_operation_broadcast({a,b},{c},{},"y0=x0+x1;",q);
        TEST(equal(c,ref,q));
    }
    {
        auto a=make_tensor<Type>(q,dp::Shape(1,2,1,3),{0,1,2,3,4,5});
        auto b=make_tensor<Type>(q,dp::Shape(2,1,3,1),{0,1,2,3,4,5,});
        auto ref=make_tensor<Type>(q,dp::Shape(2,2,3,3),{ 0,  1,  2,  1,  2,  3,  2,  3,  4,  3,  4,  5,  4,  5,  6,  5,  6,
                                                          7,  3,  4,  5,  4,  5,  6,  5,  6,  7,  6,  7,  8,  7,  8,  9,  8,
                                                          9, 10});
        dp::Tensor c(ctx,dp::Shape(2,2,3,3),a.dtype());
        std::cout << a <<"+"<<b<<"->"<<c<<std::endl;
        pointwise_operation_broadcast({a,b},{c},{},"y0=x0+x1;",q);
        TEST(equal(c,ref,q));
    }
    {
        auto a=make_tensor<Type>(q,dp::Shape(1,2,1,3,1),{0,1,2,3,4,5});
        auto b=make_tensor<Type>(q,dp::Shape(2,1,3,1,2),{0,1,2,3,4,5,6,7,8,9,10,11});
        auto ref=make_tensor<Type>(q,dp::Shape(2,2,3,3,2),{
            0,  1,  1,  2,  2,  3,  2,  3,  3,  4,  4,  5,  4,  5,  5,  6,  6,
            7,  3,  4,  4,  5,  5,  6,  5,  6,  6,  7,  7,  8,  7,  8,  8,  9,
            9, 10,  6,  7,  7,  8,  8,  9,  8,  9,  9, 10, 10, 11, 10, 11, 11,
           12, 12, 13,  9, 10, 10, 11, 11, 12, 11, 12, 12, 13, 13, 14, 13, 14,
           14, 15, 15, 16
        });
        dp::Tensor c(ctx,dp::Shape(2,2,3,3,2),a.dtype());
        std::cout << a <<"+"<<b<<"->"<<c<<std::endl;
        pointwise_operation_broadcast({a,b},{c},{},"y0=x0+x1;",q);
        TEST(equal(c,ref,q));
    }
    {
        auto a=make_tensor<Type>(q,dp::Shape(1,2,1,3,1,2),{0,1,2,3,4,5,6,7,8,9,10,11});
        auto b=make_tensor<Type>(q,dp::Shape(2,1,3,1,2,1),{0,1,2,3,4,5,6,7,8,9,10,11});
        auto ref=make_tensor<Type>(q,dp::Shape(2,2,3,3,2,2),{
                0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  2,  3,  3,  4,  4,
                5,  5,  6,  6,  7,  7,  8,  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,
                9, 10,  6,  7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 12,  8,  9,  9,
                10, 10, 11, 11, 12, 12, 13, 13, 14, 10, 11, 11, 12, 12, 13, 13, 14,
                14, 15, 15, 16,  6,  7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 12,  8,
                9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 10, 11, 11, 12, 12, 13,
                13, 14, 14, 15, 15, 16, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17,
                18, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 16, 17, 17, 18,
                18, 19, 19, 20, 20, 21, 21, 22
        });
        dp::Tensor c(ctx,dp::Shape(2,2,3,3,2,2),a.dtype());
        std::cout << a <<"+"<<b<<"->"<<c<<std::endl;
        pointwise_operation_broadcast({a,b},{c},{},"y0=x0+x1;",q);
        TEST(equal(c,ref,q));
    }
    {
        auto a=make_tensor<Type>(q,dp::Shape(1,2,1,3,1,2,1),{0,1,2,3,4,5,6,7,8,9,10,11});
        auto b=make_tensor<Type>(q,dp::Shape(2,1,3,1,2,1,2),{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23});
        auto ref=make_tensor<Type>(q,dp::Shape(2,2,3,3,2,2,2),{
                0,  1,  1,  2,  2,  3,  3,  4,  2,  3,  3,  4,  4,  5,  5,  6,  4,
                5,  5,  6,  6,  7,  7,  8,  4,  5,  5,  6,  6,  7,  7,  8,  6,  7,
                7,  8,  8,  9,  9, 10,  8,  9,  9, 10, 10, 11, 11, 12,  8,  9,  9,
                10, 10, 11, 11, 12, 10, 11, 11, 12, 12, 13, 13, 14, 12, 13, 13, 14,
                14, 15, 15, 16,  6,  7,  7,  8,  8,  9,  9, 10,  8,  9,  9, 10, 10,
                11, 11, 12, 10, 11, 11, 12, 12, 13, 13, 14, 10, 11, 11, 12, 12, 13,
                13, 14, 12, 13, 13, 14, 14, 15, 15, 16, 14, 15, 15, 16, 16, 17, 17,
                18, 14, 15, 15, 16, 16, 17, 17, 18, 16, 17, 17, 18, 18, 19, 19, 20,
                18, 19, 19, 20, 20, 21, 21, 22, 12, 13, 13, 14, 14, 15, 15, 16, 14,
                15, 15, 16, 16, 17, 17, 18, 16, 17, 17, 18, 18, 19, 19, 20, 16, 17,
                17, 18, 18, 19, 19, 20, 18, 19, 19, 20, 20, 21, 21, 22, 20, 21, 21,
                22, 22, 23, 23, 24, 20, 21, 21, 22, 22, 23, 23, 24, 22, 23, 23, 24,
                24, 25, 25, 26, 24, 25, 25, 26, 26, 27, 27, 28, 18, 19, 19, 20, 20,
                21, 21, 22, 20, 21, 21, 22, 22, 23, 23, 24, 22, 23, 23, 24, 24, 25,
                25, 26, 22, 23, 23, 24, 24, 25, 25, 26, 24, 25, 25, 26, 26, 27, 27,
                28, 26, 27, 27, 28, 28, 29, 29, 30, 26, 27, 27, 28, 28, 29, 29, 30,
                28, 29, 29, 30, 30, 31, 31, 32, 30, 31, 31, 32, 32, 33, 33, 34
        });
        dp::Tensor c(ctx,dp::Shape(2,2,3,3,2,2,2),a.dtype());
        std::cout << a <<"+"<<b<<"->"<<c<<std::endl;
        pointwise_operation_broadcast({a,b},{c},{},"y0=x0+x1;",q);
        TEST(equal(c,ref,q));
    }
    {
        auto a=make_tensor<Type>(q,dp::Shape(1,2,1,2, 1,2,1,2),{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
        auto b=make_tensor<Type>(q,dp::Shape(2,1,2,1, 2,1,2,1),{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
        auto ref=make_tensor<Type>(q,dp::Shape(2,2,2,2, 2,2,2,2),{
                0,  1,  1,  2,  2,  3,  3,  4,  2,  3,  3,  4,  4,  5,  5,  6,  4,
                5,  5,  6,  6,  7,  7,  8,  6,  7,  7,  8,  8,  9,  9, 10,  4,  5,
                5,  6,  6,  7,  7,  8,  6,  7,  7,  8,  8,  9,  9, 10,  8,  9,  9,
                10, 10, 11, 11, 12, 10, 11, 11, 12, 12, 13, 13, 14,  8,  9,  9, 10,
                10, 11, 11, 12, 10, 11, 11, 12, 12, 13, 13, 14, 12, 13, 13, 14, 14,
                15, 15, 16, 14, 15, 15, 16, 16, 17, 17, 18, 12, 13, 13, 14, 14, 15,
                15, 16, 14, 15, 15, 16, 16, 17, 17, 18, 16, 17, 17, 18, 18, 19, 19,
                20, 18, 19, 19, 20, 20, 21, 21, 22,  8,  9,  9, 10, 10, 11, 11, 12,
                10, 11, 11, 12, 12, 13, 13, 14, 12, 13, 13, 14, 14, 15, 15, 16, 14,
                15, 15, 16, 16, 17, 17, 18, 12, 13, 13, 14, 14, 15, 15, 16, 14, 15,
                15, 16, 16, 17, 17, 18, 16, 17, 17, 18, 18, 19, 19, 20, 18, 19, 19,
                20, 20, 21, 21, 22, 16, 17, 17, 18, 18, 19, 19, 20, 18, 19, 19, 20,
                20, 21, 21, 22, 20, 21, 21, 22, 22, 23, 23, 24, 22, 23, 23, 24, 24,
                25, 25, 26, 20, 21, 21, 22, 22, 23, 23, 24, 22, 23, 23, 24, 24, 25,
                25, 26, 24, 25, 25, 26, 26, 27, 27, 28, 26, 27, 27, 28, 28, 29, 29,
                30            
        });
        dp::Tensor c(ctx,dp::Shape(2,2,2,2, 2,2,2,2),a.dtype());
        std::cout << a <<"+"<<b<<"->"<<c<<std::endl;
        pointwise_operation_broadcast({a,b},{c},{},"y0=x0+x1;",q);
        TEST(equal(c,ref,q));
    }
}

void pointwise_operation_broadcast_reduce(  std::vector<dp::Tensor> xs,
                                            std::vector<dp::Tensor> ys,
                                            std::vector<double>  ws,
                                            std::string const &compute,
                                            std::string const &reduce_init,
                                            std::string const &reduce,
                                            dp::ExecutionContext const &e)
{
    using namespace dlprim;
    Context ctx(e);
    std::vector<TensorSpecs> xspec,yspec;
    std::vector<double> alpha,beta;
    for(auto const &x:xs) {
        xspec.push_back(x.specs());
    }
    for(auto const &y:ys) {
        yspec.push_back(y.specs());
        alpha.push_back(1.0);
        beta.push_back(0.0);
    }
    
    auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                        ctx,xspec,yspec,
                        ws.size(),ys[0].dtype(),compute,reduce_init,reduce);
    Tensor workspace;
    if(op->workspace() > 0)
        workspace = Tensor(ctx,Shape(op->workspace()),uint8_data);
    op->enqueue(xs,ys,workspace,ws,alpha,beta,e);
}

template<typename Type>
void test_reduce(dp::ExecutionContext const &q)
{

    dp::Context ctx(q);
    {
        auto a=make_tensor<Type>(q,dp::Shape(1),{1});
        auto b=make_tensor<Type>(q,dp::Shape(2),{2,3});
        auto ref=make_tensor<Type>(q,dp::Shape(1),{7});
        dp::Tensor c(ctx,dp::Shape(1),a.dtype());
        std::cout << a <<"+"<<b<<"->"<<c<<std::endl;
        pointwise_operation_broadcast_reduce({a,b},{c},{},"y0=x0+x1;","reduce_y0 = 0;" ,"reduce_y0 += y0;",q);
        TEST(equal(c,ref,q));
    }
    {
        auto a=make_tensor<Type>(q,dp::Shape(1,2),{0,1});
        auto b=make_tensor<Type>(q,dp::Shape(2,1),{0,1});
        auto ref=make_tensor<Type>(q,dp::Shape(1,2),{7,9});
        dp::Tensor c(ctx,dp::Shape(1,2),a.dtype());
        std::cout << a <<"+"<<b<<"->"<<c<<std::endl;
        pointwise_operation_broadcast_reduce({a,b},{c},{7},"y0=x0+w0*x1;","reduce_y0 = 0;" ,"reduce_y0 += y0;",q);
        TEST(equal(c,ref,q));
    }
    {
        auto a=make_tensor<Type>(q,dp::Shape(2,2),{1,2,7,3});
        auto ref0=make_tensor<Type>(q,dp::Shape(1,2),{7, 3});
        auto ref1=make_tensor<Type>(q,dp::Shape(1,2),{1, 1});
        dp::Tensor c0(ctx,dp::Shape(1,2),a.dtype());
        dp::Tensor c1(ctx,dp::Shape(1,2),a.dtype());
        std::cout << a <<"+"<<"->"<<c0 << "x" << c1<<","<<c1<<std::endl;
        pointwise_operation_broadcast_reduce({a},{c0,c1},{},
                    "y0=x0; y1=reduce_item;",
                    "reduce_y0 = -100; reduce_y1 = -1;" ,"if(y0 > reduce_y0) { reduce_y0 = y0; reduce_y1 = y1; }",q);
        TEST(equal(c0,ref0,q));
        TEST(equal(c1,ref1,q));
    }
    {
        auto a=make_tensor<Type>(q,dp::Shape(1,2),{0,1});
        auto b=make_tensor<Type>(q,dp::Shape(2,1),{0,1});
        auto ref0=make_tensor<Type>(q,dp::Shape(2,1),{1,15});
        auto ref1=make_tensor<Type>(q,dp::Shape(2,1),{0,56});
        dp::Tensor c0(ctx,dp::Shape(2,1),a.dtype());
        dp::Tensor c1(ctx,dp::Shape(2,1),a.dtype());
        std::cout << a <<"+"<<b<<"->"<<c0<<","<<c1<<std::endl;
        pointwise_operation_broadcast_reduce({a,b},{c0,c1},{7},"y0=x0+w0*x1; y1=y0;","reduce_y0 = 0; reduce_y1 = 1;" ,"reduce_y0 += y0; reduce_y1 *= y1;",q);
        TEST(equal(c0,ref0,q));
        TEST(equal(c1,ref1,q));
    }
    {
        auto test_eq = [&](dp::Shape as,std::vector<Type> av,dp::Shape bs,std::vector<Type> bv,dp::Shape cs,std::vector<Type> cv) 
        {
            auto a=make_tensor<Type>(q,as,av);
            auto b=make_tensor<Type>(q,bs,bv);
            auto ref=make_tensor<Type>(q,cs,cv);
            dp::Tensor c(ctx,cs,a.dtype());
            std::cout << a <<"+"<<b<<"->"<<c<<std::endl;
            pointwise_operation_broadcast_reduce({a,b},{c},{},"y0=x0+x1;","reduce_y0 = 0;" ,"reduce_y0 += y0;",q);
            TEST(equal(c,ref,q));
        };
        test_eq(dp::Shape(1,2,1),{0,1},
                dp::Shape(2,1,3),{0,1,2,3,4,5},
                dp::Shape(1,2,3),{3, 5, 7, 5, 7, 9});

        test_eq(dp::Shape(1,2,1),{0,1},
                dp::Shape(2,1,3),{0,1,2,3,4,5},
                dp::Shape(2,1,3),{ 1,  3,  5,  7,  9, 11});

        test_eq(dp::Shape(1,2,1),{0,1},
                dp::Shape(2,1,3),{0,1,2,3,4,5},
                dp::Shape(2,2,1),{3,  6, 12, 15});

        test_eq(dp::Shape(1,2,1),{0,1},
                dp::Shape(2,1,3),{0,1,2,3,4,5},
                dp::Shape(1,2,1),{15,21});

        test_eq(dp::Shape(1,2,1,2),{0,1,2,3},
                dp::Shape(2,1,3,1),{0,1,2,3,4,5},
                dp::Shape(1,2,3,2),{3,  5,  5,  7,  7,  9,  7,  9,  9, 11, 11, 13});

        test_eq(dp::Shape(1,2,1,2),{0,1,2,3},
                dp::Shape(2,1,3,1),{0,1,2,3,4,5},
                dp::Shape(1,2,1,2),{15, 21, 27, 33});

        test_eq(dp::Shape(1,2,1,2,1),{0,1,2,3},
                dp::Shape(2,1,3,1,2),{0,1,2,3,4,5,6,7,8,9,10,11},
                dp::Shape(1,2,3,2,2),{6,  8,  8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 10, 12, 12, 14, 14,
                                      16, 16, 18, 18, 20, 20, 22});

        test_eq(dp::Shape(1,2,1,2,1),{0,1,2,3},
                dp::Shape(2,1,3,1,2),{0,1,2,3,4,5,6,7,8,9,10,11},
                dp::Shape(1,2,1,2,1),{66,  78,  90, 102});

        test_eq(dp::Shape(2,1,2,1),{0,1,2,3},
                dp::Shape(2,1,3,1,2),{0,1,2,3,4,5,6,7,8,9,10,11},
                dp::Shape(2,1,2,1),{66,  78,  90, 102});

        test_eq(dp::Shape(1,2,1,2,1),{0,1,2,3},
                dp::Shape(2,1,3,1,2),{0,1,2,3,4,5,6,7,8,9,10,11},
                dp::Shape(2,1,3,1,2),{ 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50});

        test_eq(dp::Shape(2),{0,1},
                dp::Shape(1),{1},
                dp::Shape(2),{1,2});

        test_eq(dp::Shape(1,2),{0,1},
                dp::Shape(2,1),{0,1},
                dp::Shape(2,2),{0,1,1,2});

        test_eq(dp::Shape(1,2,1),{0,1},
                dp::Shape(2,1,2),{0,1,2,3},
                dp::Shape(2,2,2),{0, 1, 1, 2, 2, 3, 3, 4});

        test_eq(dp::Shape(1,2,1,2),{0,1,2,3},
                dp::Shape(2,1,2,1),{0,1,2,3},
                dp::Shape(2,2,2,2),{0, 1, 1, 2, 2, 3, 3, 4, 2, 3, 3, 4, 4, 5, 5, 6});

        test_eq(dp::Shape(1,2,1,2,1),{0,1,2,3},
                dp::Shape(2,1,2,1,2),{0,1,2,3,4,5,6,7},
                dp::Shape(2,2,2,2,2),{
                    0,  1,  1,  2,  2,  3,  3,  4,  2,  3,  3,  4,  4,  5,  5,  6,  4,
                    5,  5,  6,  6,  7,  7,  8,  6,  7,  7,  8,  8,  9,  9, 10
                });
    }
    for(size_t size : std::vector<int>({5,101,201,512,1001,2011,5099,10012,50243,100017})) {
        int C=200;
        dp::Shape as(C,size);
        dp::Shape rs(C,1);
        std::vector<Type> av(as.total_size());
        std::vector<Type> r0(rs.total_size());
        std::vector<std::int64_t> r1(rs.total_size());
        size_t pos = 0;
        int eps = 0;
        int reduced_max  = 17 * size * 5;
        if(dp::TypeTraits<Type>::data_type == dp::half_data) {
            if(reduced_max >= 2048)
                eps = std::numeric_limits<int>::max();
        }
        for(int c=0;c<C;c++) {
            for(size_t j=0;j<size;j++) {
                av.at(pos) = pos % 17;
                r0.at(c) += pos % 17 * 5;
                r1.at(c) += -(pos % 17) * 4;
                pos++;
            }
        }
        auto a=make_tensor<Type>(q,as,av);
        auto ref0=make_tensor<Type>(q,rs,r0);
        auto ref1=make_tensor<std::int64_t>(q,rs,r1);
        dp::Tensor c0(ctx,rs,a.dtype());
        dp::Tensor c1(ctx,rs,ref1.dtype());
        std::cout << a <<"->"<<c0 << "&" << c1<<std::endl;

        auto op = dp::core::PointwiseOperationBroadcastReduce::create(
            ctx,{a.specs()},{c0.specs(),c1.specs()},
            0,dp::float_data,
            "y0=x0; y1=-x0;",
            "reduce_y0 = 0; reduce_y1 = 0;" ,
            "reduce_y0 += y0; reduce_y1 += y1;");
        dp::Tensor ws;
        if(op->workspace() > 0)
            ws = dp::Tensor(ctx,dp::Shape(op->workspace()),dp::uint8_data);

        op->enqueue({a},{c0,c1},ws,{},{1,2},{0,0},q);
        op->enqueue({a},{c0,c1},ws,{},{1,2},{4,1},q);

        TEST(equal(c0,ref0,q,eps));
        TEST(equal(c1,ref1,q,eps));
    }
    for(int b : std::vector<int>{2,5,64}) {
        for(int hw : std::vector<int>{7,20,37}) {
            int C=500;
            dp::Shape as(b,C,hw,hw+1);
            dp::Shape rs(C,1,1);
            std::vector<Type> av(as.total_size());
            std::vector<Type> rv(rs.total_size());
            size_t pos = 0;
            for(int i=0;i<b;i++) {
                for(int c=0;c<C;c++) {
                    for(int y=0;y<hw;y++) {
                        for(int x=0;x<hw+1;x++) {
                            av.at(pos) = pos % 7;
                            rv.at(c) += pos % 7;
                            pos++;
                        }
                    }
                }
            }
            auto a=make_tensor<Type>(q,as,av);
            auto ref=make_tensor<Type>(q,rs,rv);
            dp::Tensor c(ctx,rs,a.dtype());
            std::cout << a <<"->"<<c<<std::endl;
            pointwise_operation_broadcast_reduce({a},{c},{},"y0=x0;","reduce_y0 = 0;" ,"reduce_y0 += y0;",q);
            int eps = 0;
            if(dp::TypeTraits<Type>::data_type == dp::half_data && (b*hw*hw*7) >= 2048) {
                eps = std::numeric_limits<int>::max();
            }
            TEST(equal(c,ref,q,eps));
        }
    }
    for(int b : std::vector<int>{2,5}) {
        for(int hw : std::vector<int>{7,20,37}) {
            dp::Shape as(hw,b,hw+1,2*b,hw-1);
            dp::Shape rs(1,b,1,2*b,1);
            std::vector<Type> av(as.total_size());
            std::vector<Type> rv(rs.total_size());
            size_t pos = 0;
            for(int i=0;i<hw;i++) {
                for(int j=0;j<b;j++) {
                    for(int k=0;k<hw+1;k++) {
                        for(int l=0;l<b*2;l++) {
                            for(int n=0;n<hw-1;n++) {
                                av.at(pos) = pos % 17;
                                rv.at(j*2*b+l) += pos % 17;
                                pos++;
                            }
                        }
                    }
                }
            }
            auto a=make_tensor<Type>(q,as,av);
            auto ref=make_tensor<Type>(q,rs,rv);
            dp::Tensor c(ctx,rs,a.dtype());
            std::cout << a <<"->"<<c<<std::endl;
            pointwise_operation_broadcast_reduce({a},{c},{},"y0=x0;","reduce_y0 = 0;" ,"reduce_y0 += y0;",q);
            int eps = 0;
            if(dp::TypeTraits<Type>::data_type == dp::half_data ) {
                eps = std::numeric_limits<int>::max();
            }
            TEST(equal(c,ref,q,eps));
        }
    }

}


int main(int argc,char **argv)
{
    if(argc!=2) {
        std::cerr << "Use paltform:device" << std::endl;
        return 1;
    }
    try {

        std::cout << "Testing shape" << std::endl;
        test_shape();

        dp::Context ctx(argv[1]);
        bool with_half = ctx.check_device_extension("cl_khr_fp16");
        if(ctx.is_cpu_context()) {
            std::cout << "CPU - exit" << std::endl;
            return 0;
        }
        dp::ExecutionContext q = ctx.make_execution_context();
        std::cout << ctx.name() << std::endl;
        
        std::cout << "Pointwise" << std::endl;
        test_pointwise<float>(q);
        test_pointwise<int>(q);
        test_pointwise<int64_t>(q);
        test_pointwise<int16_t>(q);
        if(with_half)
            test_pointwise<my_half>(q);
        std::cout << "Broadcast" << std::endl;
        test_broadcast<float>(q);
        test_broadcast<int>(q);
        test_broadcast<int64_t>(q);
        test_broadcast<int16_t>(q);
        test_broadcast<uint8_t>(q);
        if(with_half)
            test_broadcast<my_half>(q);
        std::cout << "Broadcast Reduce" << std::endl;
        test_reduce<float>(q);
        test_reduce<int>(q);
        if(with_half)
            test_reduce<my_half>(q);
    }
    catch(std::exception const &e) {
        std::cerr <<"Failed:"<< e.what() << std::endl;
        return 1;
    }
    return 0;

}
