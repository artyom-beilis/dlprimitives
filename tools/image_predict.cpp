#include <dlprim/net.hpp>
#include <dlprim/json.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

namespace dp = dlprim;


struct Config {
    std::vector<float> fact;
    std::vector<float> off;
    std::vector<std::string> class_names;
    int c=-1,h=-1,w=-1;
    int classes=-1;
    void load(std::string const &path)
    {
        if(path.empty()) {
            fact.resize(c,1/255.0f);
            off.resize(c,0.0f);
            return;
        }
        std::ifstream f(path);
        dp::json::value v;
        if(!v.load(f,true)) {
            throw std::runtime_error("Failed to read: " + path);
        }
        if(!v.find("mean").is_undefined()) {
            auto mean = v.get<std::vector<float> >("mean");
            auto std = v.get<std::vector<float> >("std");
            fact.resize(mean.size());
            off.resize(mean.size());
            for(unsigned i=0;i<mean.size();i++) {
                fact[i] = 1.0 / 255.0f / std[i];
                off[i] = -mean[i] / std[i];
            }
            if(int(fact.size()) != c) {
                std::cerr << "Norm " << fact.size() << " exp " << c << std::endl;
                throw std::runtime_error("Invalid normalization size");
            }
        }
        if(!v.find("class_names").is_undefined()) {
            class_names = v.get<std::vector<std::string> >("class_names");
            if(int(class_names.size()) != classes) {
                throw std::runtime_error("Names of classes does not match network");
            }
        }
    }
};

void load_image(dp::Tensor &t,int batch,std::string const &path,Config const &cfg)
{
    float *img = t.data<float>() + t.shape().size_no_batch() * batch;
    std::ifstream f(path,std::ifstream::binary);
    std::string header;
    f >> header;
    if(header == "P5") {
        if(cfg.c != 1)
            throw std::runtime_error("PGM requires 1 channel: " + path);
    }
    else if(header == "P6") {
        if(cfg.c != 3) {
            throw std::runtime_error("PPM requires 3 channel: " + path);
        }
    }
    else {
        throw std::runtime_error("Unspoorted format " + path);
    }
    std::string data,line;
    int w=-1,h=-1,bits=-1;
    while(std::getline(f,line)) {
        if(line.c_str()[0] == '#')
            continue;
        data += ' ';
        data += line;
        std::istringstream ss(data);
        w=h=bits = -1;
        ss >> w >> h >> bits;
        if(bits != -1)
            break;
    }
    if(bits != 255 || w <=0 || h <= 0) {
        throw std::runtime_error("Failed to parse " + path);
    }
    std::vector<unsigned char> buf(w*h*cfg.c);
    f.read(reinterpret_cast<char*>(buf.data()),buf.size());
    if(!f) {
        throw std::runtime_error("Failed to read " + path);
    }
    if(w < cfg.w || h < cfg.h) {
        throw std::runtime_error("image too small " + path);
    }
    int dc = (w - cfg.w)/2;
    int dr = (h - cfg.h)/2;
    for(int chan = 0;chan < cfg.c ;chan ++) {
        for(int r=0;r<cfg.h;r++) {
            for(int c=0;c<cfg.w;c++) {
                float v = cfg.fact[chan] * buf[(r+dr) * w * cfg.c + (c+dc) * cfg.c + chan] + cfg.off[chan];
                *img++ = v;
            }
        }
    }
}
int argmax(float *p,int n)
{
    int r = 0;
    for(int i=1;i<n;i++){
        if(p[i] > p[r])
            r = i;
    }
    return r;
}

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
        if(argc<5) {
            std::cerr << "Usage [-oconfig.json ] device net.json net.h5 img1.ppm [ img2.ppm ]..." << std::endl;
            return 1;
        }
        dp::Context ctx(argv[1]);
        std::cerr << "Using: " << ctx.name() << std::endl;
        
        std::string net_js = argv[2];
        std::string net_h5 = argv[3];

        dp::Net net(ctx);
        net.mode(dp::CalculationsMode::predict);
        net.load_from_json_file(net_js);
        net.setup();
        net.load_parameters_from_hdf5(net_h5);
        net.copy_parameters_to_device();
        dp::Tensor data = net.input(0),prob = net.output(0);

        Config cfg;
        int batch = data.shape()[0];
        cfg.c = data.shape()[1];
        cfg.h = data.shape()[2];
        cfg.w = data.shape()[3];
        cfg.classes = prob.shape()[1];

        cfg.load(config);

       
        cl::CommandQueue queue(ctx.make_queue());
        dp::ExecutionContext q(queue);
        std::vector<std::string> names;
        int n = 0;
        std::ofstream rep("report_dp.csv");
        for(int i=4;i<argc;i++) {
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
