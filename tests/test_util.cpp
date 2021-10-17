#include <dlprim/core/util.hpp>
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
                                            float_data,q);
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
                                            float_data,q);
                x.to_host(q,xv);
                TEST(memcmp(xv,xv2,sizeof(float)*8) == 0);
            }
            {
                for(int dims=2;dims<=5;dims++) {
                    int sizes[5]={2,3,5,7,11};
                    Shape s=Shape::from_range(sizes+0,sizes+dims);
                    Shape strides=s;
                    int scale=1;
                    for(int i=dims-1;i>=0;i--) {
                        strides[i]=scale;
                        scale *= s[i];
                    }
                    std::cout << "Checking " << s <<"/"<<strides << std::endl;
                    std::vector<int> vals_x(s.total_size());
                    std::vector<int> vals_y(s.total_size());
                    for(unsigned i=0;i<vals_x.size();i++)
                        vals_x[i]=i+5;
                    Tensor x(ctx,s);
                    Tensor y(ctx,s);
                    x.to_device(q,vals_x.data());
                    core::copy_strided(s,x.device_buffer(),x.device_offset(),strides,
                                         y.device_buffer(),y.device_offset(),strides,
                                         int32_data,q);
                    y.to_host(q,vals_y.data());
                    TEST(vals_x == vals_y);
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
