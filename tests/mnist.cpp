///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/net.hpp>
#include <fstream>
#include <iostream>

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
            static int count;
            count ++;
            if(count < 50) {
                std::ofstream tmp("digit_" + std::to_string(count) + "_" + std::to_string(int(lbl)) + ".pgm");
                tmp << "P5\n28 28\n255\n";
                tmp.write((char*)(buf),img_size);
                tmp.close();
            }
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
    bool enable_profiling = false;
    if(argc >= 2 && argv[1]==std::string("-t")) {
        enable_profiling = true;
        argv++;
        argc--;
    }
    if(argc<6) {
        std::cerr << "Usage [-t] device net.json net.h5 minst_data mnist_labels" << std::endl;
        return 1;
    }
    dp::Context ctx(argv[1]);
    std::cout << "Using: " << ctx.name() << std::endl;
    
    std::string net_js = argv[2];
    std::string net_h5 = argv[3];
    std::string mnist_img = argv[4];
    std::string mnist_lbl = argv[5];

    MnistReader reader(mnist_img,mnist_lbl);
    dp::Net net(ctx);
    net.load_from_json_file(net_js);
    net.setup();
    net.load_parameters(net_h5);
    net.copy_parameters_to_device();
    dp::Tensor data = net.input(0);
    dp::Tensor prob = net.output(0);
    int batch = data.shape()[0];
    std::vector<int> labels(batch);
    int n;

    cl::CommandQueue queue=ctx.make_queue(enable_profiling ? CL_QUEUE_PROFILING_ENABLE : 0);
    std::shared_ptr<dp::TimingData> timing;
    dp::ExecutionContext q(queue);
    if(enable_profiling) {
        timing.reset(new dp::TimingData);
        q.enable_timing(timing);
    }
    int correct = 0;
    int total = 0;
    int total_batches = 0;
    double total_time = 0;
    bool reported = false;
    while((n = reader.get_batch(labels.data(),data.data<float>(),batch)) > 0) {
        if(n != batch) {
            data.reshape(dp::Shape(n,1,28,28));
            net.reshape();
        }
        if(timing)
            timing->reset();
        auto start = std::chrono::system_clock::now();
        data.to_device(q,false);
        net.forward(q);
        prob.to_host(q,true);
        auto stop = std::chrono::system_clock::now();
        if(total != 0 && !reported && timing) {
            double total_event_time = 0;
            for(auto &d : timing->events()) {
                try {
                    auto end =   d->event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                    auto start = d->event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                    double time = (end - start) * 1e-3;
                    total_event_time += time;
                    int s = d->section;
                    std::stack<char const *> sections;
                    while(s!=-1) {
                        auto &sec = timing->sections().at(s);
                        sections.push(sec.name);
                        s=sec.parent;
                    }
                    while(!sections.empty()) {
                        std::cout << sections.top() << ":";
                        sections.pop();
                    }
                    std::cout << d->name;
                    if(d->index != -1)
                        std::cout << '[' << d->index << ']';
                    std::cout << "  " << time << " us" << std::endl;
                }
                catch(cl::Error const &e) {
                    std::cerr << "Failed for " << d->name << " " << e.what() << e.err() << std::endl;
                }
            }
            std::cout << "Total GPU " << total_event_time << " us , real " << std::chrono::duration_cast<std::chrono::duration<double> > ((stop-start)).count() * 1e6 << " us" << std::endl;
            reported = true;
        }
        total_time += std::chrono::duration_cast<std::chrono::duration<double> > ((stop-start)).count();
        total_batches ++;
        for(int i=0;i<n;i++) {
            int pred = argmax(prob.data<float>() + 10*i,10);
            #ifdef PRINT_LOG
                for(int r=0;r<28;r++) {
                    for(int c=0;c<28;c++) {
                        std::cout << (data.data<float>()[28*28*i + 28*r + c] > 0.2 ? 'X' : ' ');
                        //std::cout << (data.data<float>()[28*28*i + 28*r + c]) << " ";
                    }
                    std::cout << std::endl;
                }
                for(int j=0;j<10;j++)
                    std::cout << prob.data<float>()[10*i + j] << " ";
                std::cout << "->" << pred << " exp " << labels[i] << std::endl;
            #endif            
            if(pred == labels[i])
                correct++;
        }
        total += n;
        if(n < batch)
            break;
    }
    std::cout << "Correct " << correct << " out of " << total << " = " << (100.0 * correct / total) <<"%" << std::endl;
    std::cout << "Time per sample: " << (total_time / total * 1e6) << "us" << std::endl;
    std::cout << "Time per batch:  " << (total_time / total_batches * 1e6) << "us" << std::endl;

}
