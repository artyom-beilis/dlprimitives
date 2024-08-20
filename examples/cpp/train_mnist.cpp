///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/net.hpp>
#include <dlprim/solvers/sgd.hpp>
#include <dlprim/solvers/adam.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

namespace dp = dlprim;

class MnistReader {
public:
    MnistReader(std::string const &img,std::string const &lbl)
    {
        img_.open(img,std::fstream::binary);
        lbl_.open(lbl,std::fstream::binary);
        if(!img_ || !lbl_) {
            throw std::runtime_error("Failed to open mnsit db");
        }
        char tmp[16];
        lbl_.read(tmp,8);
        img_.read(tmp,16);
    }
    void rewind()
    {
        img_.clear();
        img_.seekg(16);
        lbl_.clear();
        lbl_.seekg(8);
    }
    int get_batch(int *lables,float *data,int max)
    {
        int read = 0;
        constexpr int img_size = 28*28;
        constexpr float factor = 1.0f/255.0f;
        unsigned char buf[img_size];
        char lbl;
        while(max > 0 && !img_.eof()) {
            img_.read((char*)buf,img_size);
            lbl_.read(&lbl,1);
            if(img_.gcount()!=img_size || lbl_.gcount() != 1)
                break;
            for(int i=0;i<img_size;i++) {
                *data++ = buf[i] * factor;
            }
            *lables++ = lbl;
            max--;
            read++;
        }
        return read;
    }

private:
    std::ifstream img_,lbl_;
};


int main(int argc,char **argv)
{
    if(argc<5) {
        std::cerr << "Usage device net.json mnist_img mnist_lbl" << std::endl;
        return 1;
    }
    dp::Context ctx(argv[1]);
    std::cout << "Using: " << ctx.name() << std::endl;
    dp::ExecutionContext q=ctx.make_execution_context();
    
    std::string net_js = argv[2];
    std::string mnist_img = argv[3];
    std::string mnist_lbl = argv[4];

    MnistReader reader(mnist_img,mnist_lbl);
    dp::Net net(ctx);
    net.mode(dp::CalculationsMode::train);
    net.load_from_json_file(net_js);
    net.setup();
    net.initialize_parameters(q);
    
    dp::Tensor data = net.tensor("data");
    dp::Tensor labels  = net.tensor("label");
    dp::Tensor loss = net.tensor("loss");
    dp::Tensor loss_diff = net.tensor_diff("loss");

    int batch = data.shape()[0];

    //dp::solvers::SGD sgd(ctx);
    dp::solvers::Adam sgd(ctx);
    sgd.init(net,q);

    std::cout << "Start training" << std::endl;

    for(int epoch = 0;epoch < 5;epoch ++) {
        auto start = std::chrono::high_resolution_clock::now();
        float total_loss = 0;
        int total = 0;
        int n;
        int bc=0;
        float acc = 0;
        while((n = reader.get_batch(labels.data<int>(),data.data<float>(),batch)) > 0) {
            bc++;
            if(n != int(data.shape()[0])) {
                data.reshape(dp::Shape(n,1,28,28));
                labels.reshape(dp::Shape(n));
                net.reshape();
            }

            data.to_device(q,false);
            labels.to_device(q,false);
            sgd.zero_grad(net,q);
            net.forward(q);
            net.backward(q,true);
            sgd.apply(net,q);
            net.tensor("fc2").to_host(q,false);
            loss.to_host(q,true);
            total_loss += n * loss.data<float>()[0];
            total += n;
            float *ptr = net.tensor("fc2").data<float>();
            for(int j=0;j<n;j++,ptr+=10) {
                int argmax = 0;
                for(int k=1;k<10;k++)
                    if(ptr[k] > ptr[argmax])
                        argmax = k;
                acc += float(argmax == labels.data<int>()[j]);
            }
            //if(bc == 100 && epoch == 0)
            //    sgd.lr *= 0.1;
            if((bc+1) % 100 == 0)
                std::cout << "Epoch/iter " << epoch << "/" << (bc+1) << " loss=" << total_loss / total << " acc="<< acc /total << std::endl;
        }
        std::cout << "Epoch/iter " << epoch << "/" << bc << " loss=" << total_loss / total << " acc="<< acc /total << std::endl;
        reader.rewind();
        //if(epoch <= 2)
        //    sgd.lr *= 0.1;
        net.copy_parameters_to_host();
        net.save_parameters("snap_" + std::to_string(epoch+1) + ".dlp");
        auto end = std::chrono::high_resolution_clock::now();
        auto passed = std::chrono::duration_cast<std::chrono::duration<double> > ((end-start)).count();
        std::cout << "Executed in " << passed << " second" << std::endl;
    }

}
