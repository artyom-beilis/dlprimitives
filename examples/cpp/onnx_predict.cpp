///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/net.hpp>
#include <dlprim/onnx.hpp>
#include "util.hpp"
namespace dp = dlprim;


int main(int argc,char **argv)
{
    try {
        std::string config;
        while(argc >= 2 && argv[1][0] == '-') {
            std::string flag = argv[1];
            if(flag.substr(0,2) == "-o") {
                config = flag.substr(2);
            }
            else {
                std::cerr << "Invalid Flag " << flag << std::endl;
                return 1;
            }
            argv++;
            argc--;
        }
        if(argc<4) {
            std::cerr << "Usage [-oconfig.json ] device net.onnx img1.ppm [ img2.ppm ]..." << std::endl;
            return 1;
        }
        dp::Context ctx(argv[1]);
        std::cerr << "Using: " << ctx.name() << std::endl;
        
        std::string onnx_path = argv[2];

        dp::ONNXModel model;
        model.load(onnx_path);
        
        dp::Net net(ctx);
        net.load_model(model);
        dp::Tensor data = net.input(0),prob = net.output(0);

        Config cfg;
        int batch = data.shape()[0];
        cfg.c = data.shape()[1];
        cfg.h = data.shape()[2];
        cfg.w = data.shape()[3];
        cfg.classes = prob.shape()[1];

        cfg.load(config);

       
        dp::ExecutionContext q = ctx.make_execution_context();
        std::vector<std::string> names;
        int n = 0;
        std::ofstream rep("report_dp.csv");
        for(int i=3;i<argc;i++) {
            load_image(data,n,argv[i],cfg);
            names.push_back(argv[i]);
            n++;
            if(n<batch && i+1<argc)
                continue;
            if(n != batch) {
                data.reshape(dp::Shape(n,cfg.c,cfg.h,cfg.w));
                net.reshape();
            }
            data.to_device(q,true);
            net.forward(q);
            prob.to_host(q,true);
            for(int j=0;j<n;j++) {
                float *probv = prob.data<float>() + cfg.classes*j;
                int pred = argmax(probv,cfg.classes);
                std::cout << names[j] << "," << pred << ",";
                rep << names[j] << "," << pred << ",";
                if(cfg.class_names.empty()) {
                    std::cout << "class_" << pred; 
                    rep <<  "class_" << pred;
                }
                else {
                    std::cout << cfg.class_names[pred];
                    rep << cfg.class_names[pred];
                }
                for(int k=0;k<cfg.classes;k++) {
                    if(k < 5)
                        std::cout << "," << probv[k];
                    rep << "," << probv[k];
                }
                std::cout << std::endl;
                rep << std::endl;
            }
            n=0;
            names.clear();
        }
    }
    catch(cl::Error const &e) {
        std::cerr << "OpenCL error:" << e.what() << " " << e.err() << std::endl;
        return 1;
    }
    catch(std::exception const &ex) {
        std::cerr << "Error:" << ex.what() << std::endl;
        return 1;
    }

}
