#include <dlprim/net.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace dp = dlprim;

void xavier(dp::Tensor &t)
{
    float *p = t.data<float>();
    int no = t.shape()[0];
    int ni = t.shape().size_no_batch();
    int total = t.shape().total_size();
    float factor = 1/std::sqrt(float(no+ni));
    for(int i=0;i<total;i++) {
        p[i] = factor*(float(rand())/RAND_MAX - 0.5f);
    }
}

void xavier(dp::Net &n)
{
    for(auto &p : n.params()) {
        xavier(p.second);
    }
}

int main(int argc,char **argv)
{
    try {
        bool enable_profiling = false;
        bool enable_backward = false;
        bool force_cpu_times = false;
        int warm = 5;
        int iters = 20;
        while(argc >= 2 && argv[1][0] == '-') {
            std::string flag = argv[1];
            if(flag == "-t") {
                enable_profiling = true;
            }
            else if(flag == "-b") 
                enable_backward = true;
            else if(flag == "-C")
                force_cpu_times = true;
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
        if(argc<3) {
            std::cerr << "Usage [-t] [-iNNN] [-wMMM] device net.json [net.h5]" << std::endl;
            return 1;
        }
        dp::Context ctx(argv[1]);
        std::cout << "Using: " << ctx.name() << std::endl;
        
        std::string net_js = argv[2];
        std::string net_h5 = argc >= 4 ? argv[3] : "";

        dp::Net net(ctx);
        if(enable_backward)
            net.mode(dp::CalculationsMode::train);
        net.load_from_json_file(net_js);
        net.setup();
        if(!net_h5.empty()) {
            net.load_parameters_from_hdf5(net_h5);
        }
        else {
            xavier(net);
        }
        net.copy_parameters_to_device();
        net.reshape();
        std::vector<dp::Tensor> data,res,res_diff;
        std::cout << "Inputs" << std::endl;
        for(unsigned i=0;i<net.input_names().size();i++) {
            std::cout << "- " << net.input_names()[i] <<": " << net.input(i).shape() << std::endl;
            data.push_back(net.input(i));
        }
        std::cout << "Outputs" << std::endl;
        for(unsigned i=0;i<net.output_names().size();i++) {
            std::cout <<"- " << net.output_names()[i] << ": "<< net.output(i).shape() << std::endl;
            res.push_back(net.output(i));
            if(enable_backward)
                res_diff.push_back(net.tensor_diffs()[net.output_names()[i]]);
        }
        
        cl::CommandQueue queue=ctx.make_queue((enable_profiling && !force_cpu_times)? CL_QUEUE_PROFILING_ENABLE : 0);
        std::shared_ptr<dp::TimingData> timing;
        dp::ExecutionContext q(queue);
        if(enable_profiling) {
            timing.reset(new dp::TimingData);
            timing->cpu_only = force_cpu_times;
            q.enable_timing(timing);
        }
        int total = 0;
        int total_batches = 0;
        double total_time = 0;
        double fw_time = 0;
        double bw_time = 0;
        for(auto &t : data) {
            size_t total = t.shape().total_size();
            if(t.dtype()==dp::float_data) {
                float *ptr = t.data<float>();
                for(size_t j=0;j<total;j++) 
                    ptr[j] = float(rand())/RAND_MAX;
            }
            else if(t.dtype()==dp::int32_data) {
                int *ptr = t.data<int>();
                for(size_t j=0;j<total;j++) 
                    ptr[j] = 2*(float(rand())/RAND_MAX);
            }
            else {
                throw std::runtime_error("Unsuported data");
            }
        }
        for(int i=-warm;i<iters;i++) {
            if(timing)
                timing->reset();
            auto start = std::chrono::high_resolution_clock::now();
            auto bw_point = start;
            for(size_t j=0;j<data.size();j++) {
                dp::ExecGuard g(q,"to_device");
                data[j].to_device(q,force_cpu_times);
            }
            net.forward(q,force_cpu_times);
            for(size_t j=0;j<res.size();j++) {
                dp::ExecGuard g(q,"to_host_results");
                res[j].to_host(q,force_cpu_times || (j+1 == res.size()));
            }
            if(enable_backward) {
                bw_point = std::chrono::high_resolution_clock::now();
                for(size_t j=0;j<res_diff.size();j++) {
                    if(res_diff[j].shape().total_size() == 1) {
                        dp::ExecGuard g(q,"to_device_loss");
                        res_diff[j].data<float>()[0] = 1.0;
                        res_diff[j].to_device(q,force_cpu_times);
                    }
                }
                net.backward(q,force_cpu_times);
                if(!ctx.is_cpu_context())
                    q.queue().finish();
            }
            auto stop = std::chrono::high_resolution_clock::now();
            auto passed = std::chrono::duration_cast<std::chrono::duration<double> > ((stop-start)).count();
            std::cout << "Step " <<std::fixed << std::setw(2) << i << " " << std::setw(10) << std::setprecision(3) <<  passed * 1e3;
            if(enable_backward) {
                auto fw = std::chrono::duration_cast<std::chrono::duration<double> > ((bw_point-start)).count();
                auto bw = std::chrono::duration_cast<std::chrono::duration<double> > ((stop-bw_point)).count();
                std::cout << std::setw(10) << fw*1e3 << std::setw(10) << bw * 1e3;
                if(i>=0) {
                    fw_time += fw;
                    bw_time += bw;
                }
            }
            std::cout << std::endl;
            if(i == 0 && timing) {
                double total_event_time = 0;
                if(ctx.is_cpu_context()||force_cpu_times) {
                    for(unsigned i=0;i<timing->sections().size();i++) {
                        int s = i;
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
                        std::cout <<" " << timing->sections()[i].time_sec * 1e3 << " ms" << std::endl;
                    }
                }
                else {
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
            }
            if(i>=0) {
                total_time += passed;
                total_batches ++;
                total += data[0].shape()[0];
            }
        }
        std::cout << "Time per sample: " << (total_time / total * 1e3) << " ms" << std::endl;
        std::cout << "Time per batch:  " << (total_time / total_batches * 1e3) << " ms" << std::endl;
        if(enable_backward) {
            std::cout << "FW time per batch:  " << (fw_time / total_batches * 1e3) << " ms" << std::endl;
            std::cout << "BW time per batch:  " << (bw_time / total_batches * 1e3) << " ms" << std::endl;
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
