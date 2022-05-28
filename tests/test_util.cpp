///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/util.hpp>
#include <dlprim/core/pointwise.hpp>
#include "test.hpp"
#include <iostream>

int main(int argc,char **argv)
{
    if(argc!=2) {
        std::cerr << "test_util device" << std::endl;
        return 1;
    }
    try {
        using namespace dlprim;
        Context ctx(argv[1]);
        ExecutionContext q=ctx.make_execution_context();
        if(ctx.is_cpu_context()) {
            std::cout << "No need test for cpu" << std::endl;
            return 0;
        }
        { 
            {
                std::cout << "Test 1.1" << std::endl;
                float xv[8]={1,2,3,4,5,6,7,8};
                float xv2[8]={1.5,2,3.5,4,5.5,6,7.5,8};
                Tensor x(ctx,Shape(8));
                Tensor y(ctx,Shape(4));
                float yv[4];
                x.to_device(q,xv);
                core::copy_strided(Shape(4),x.device_buffer(),x.device_offset(),Shape(2),
                                            y.device_buffer(),y.device_offset(),Shape(1),
                                            float_data,float_data,q);
                y.to_host(q,yv);
                TEST(yv[0]==1);
                TEST(yv[1]==3);
                TEST(yv[2]==5);
                TEST(yv[3]==7);
                std::cout << "Test 1.2" << std::endl;
                yv[0] = 1.5;
                yv[1] = 3.5;
                yv[2] = 5.5;
                yv[3] = 7.5;
                y.to_device(q,yv);
                core::copy_strided(Shape(4),y.device_buffer(),y.device_offset(),Shape(1),
                                            x.device_buffer(),x.device_offset(),Shape(2),
                                            float_data,float_data,q);
                x.to_host(q,xv);
                TEST(memcmp(xv,xv2,sizeof(float)*8) == 0);
            }
            {
                for(int dims=1;dims<=dlprim::max_tensor_dim;dims++) {
                    for(int strides_mask = 0;strides_mask < (1<<dlprim::max_tensor_dim);strides_mask++) {
                        int sizes[dlprim::max_tensor_dim]={2,3,5,7,11,13,17,19};
                        Shape s=Shape::from_range(sizes+0,sizes+dims);
                        Shape src_s = s;
                        Shape strides_src=s,strides_tgt=s;
                        int scale_src=1;
                        int scale_tgt=1;
                        for(int i=dims-1;i>=0;i--) {
                            strides_src[i]=scale_src;
                            strides_tgt[i]=scale_tgt;
                            if(strides_mask & (1<<i)) {
                                scale_src *= s[i];
                            }
                            else {
                                src_s[i] = 1;
                                strides_src[i] = 0;
                            }
                            scale_tgt *= s[i];
                        }
                        std::cout << "Checking " << src_s << " as "<< s <<"/"<<strides_src << "->" << s << "/"<<strides_tgt<< std::endl;
                        std::vector<int> vals_x(src_s.total_size());
                        std::vector<float> vals_y(s.total_size());
                        std::vector<float> vals_y_ref(s.total_size());
                        for(unsigned i=0;i<vals_x.size();i++)
                            vals_x[i]=i+5;
                        Tensor x(ctx,src_s,int32_data);
                        Tensor y_ref(ctx,s,float_data);
                        Tensor y(ctx,s,float_data);
                        x.to_device(q,vals_x.data());
                        core::pointwise_operation_broadcast({x},{y_ref},{},"y0=x0;",q);
                        core::copy_strided(s,x.device_buffer(),x.device_offset(),strides_src,
                                             y.device_buffer(),y.device_offset(),strides_tgt,
                                             int32_data,float_data,q);
                        y.to_host(q,vals_y.data());
                        y_ref.to_host(q,vals_y_ref.data());
                        if(vals_y_ref != vals_y) {
                            for(size_t i=0;i<vals_y_ref.size();i++) {
                                std::cerr<< i <<" " << vals_y_ref[i] << " " << vals_y[i] << std::endl;
                            }
                        }
                        TEST(vals_y_ref == vals_y);
                    }
                }
            }
        }
    }
    catch(std::exception const &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
