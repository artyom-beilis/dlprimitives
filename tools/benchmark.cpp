#include <dlprim/net.hpp>
#include <fstream>
#include <iostream>

namespace dp = dlprim;

int main(int argc,char **argv)
{
    try {
        bool enable_profiling = false;
        int warm = 10;
        int iters = 100;
        while(argc >= 2 && argv[1][0] == '-') {
            std::string flag = argv[1];
            if(flag == "-t") {
                enable_profiling = true;
            }
            else if(flag.substr(0,2) == "-i" && flag.size() > 2) {
                iters = atoi(flag.c_str()+2);
            }
            else if(flag.substr(0,2) == "-w" && flag.size() > 2) {
                warm = atoi(flag.c_str()+2);
            }
            else {
                std::cerr << "Invalid Flag " << flag << std::endl;
                return 1;
            }
            argv++;
            argc--;
        }
        if(argc<4) {
            std::cerr << "Usage [-t] [-iNNN] [-wMMM] device net.json net.h5" << std::endl;
            return 1;
        }
        dp::Context ctx(argv[1]);
        std::cout << "Using: " << ctx.name() << std::endl;
        
        std::string net_js = argv[2];
        std::string net_h5 = argv[3];

        dp::Net net(ctx);
        net.load_from_json_file(net_js);
        net.setup();
        net.load_parameters_from_hdf5(net_h5);
        net.copy_parameters_to_device();
        std::vector<dp::Tensor> data,res;
        std::cout << "Inputs" << std::endl;
        for(unsigned i=0;i<net.input_names().size();i++) {
            std::cout << "- " << net.input_names()[i] <<": " << net.input(i).shape() << std::endl;
            data.push_back(net.input(i));
        }
        std::cout << "Outputs" << std::endl;
        for(unsigned i=0;i<net.output_names().size();i++) {
            std::cout <<"- " << net.output_names()[i] << ": "<< net.output(i).shape() << std::endl;
            res.push_back(net.output(i));
        }
        
        cl::CommandQueue queue=ctx.make_queue(enable_profiling ? CL_QUEUE_PROFILING_ENABLE : 0);
        std::shared_ptr<dp::TimingData> timing;
        dp::ExecutionContext q(queue);
        if(enable_profiling) {
            timing.reset(new dp::TimingData);
            q.enable_timing(timing);
        }
        int total = 0;
        int total_batches = 0;
        double total_time = 0;
        for(auto &t : data) {
            size_t total = t.shape().total_size();
            float *ptr = t.data<float>();
            for(size_t j=0;j<total;j++) 
                ptr[j] = float(rand())/RAND_MAX;
        }
        for(int i=-warm;i<iters;i++) {
            if(timing)
                timing->reset();
            auto start = std::chrono::system_clock::now();
            for(size_t j=0;j<data.size();j++)
                data[j].to_device(q,false);
            net.forward(q);
            for(size_t j=0;j<res.size();j++)
                res[j].to_host(q,j+1 == res.size());
            auto stop = std::chrono::system_clock::now();
            if(i == 0 && timing) {
                double total_event_time = 0;
                for(auto &d : timing->events()) {
                    try {
                        auto end =   d->event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                        auto start = d->event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                        double time = (end - start) * 1e-6;
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
                        std::cout << "  " << time << " ms" << std::endl;
                    }
                    catch(cl::Error const &e) {
                        std::cerr << "Failed for " << d->name << " " << e.what() << e.err() << std::endl;
                    }
                }
                std::cout << "Total GPU " << total_event_time << " ms , real " << std::chrono::duration_cast<std::chrono::duration<double> > ((stop-start)).count() * 1e3 << " ms" << std::endl;
            }
            if(i>=0) {
                total_time += std::chrono::duration_cast<std::chrono::duration<double> > ((stop-start)).count();
                total_batches ++;
            }
            total += data[0].shape()[0];
        }
        std::cout << "Time per sample: " << (total_time / total * 1e3) << "ms" << std::endl;
        std::cout << "Time per batch:  " << (total_time / total_batches * 1e3) << "ms" << std::endl;
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
