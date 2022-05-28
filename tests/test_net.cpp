///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/net.hpp>
#include <dlprim/json.hpp>
#include <dlprim/solvers/adam.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace dp = dlprim;
using dp::Tensor;

bool compare_tensors(std::string const &name,Tensor &act,Tensor &ref,float eps = 1e-5)
{
    std::cout << "Testing " << name << std::endl;
    float *a=act.data<float>();
    float *r=ref.data<float>();
    size_t size = ref.shape().total_size();
    float sum=0,sum2=0;
    for(size_t j=0;j<size;j++) {
        sum +=r[j];
        sum2+=r[j]*r[j];
    }
    sum /= size;
    sum2 /= size;
    float variance = sum2 - sum*sum;
    float dev = std::sqrt(variance);
    if(dev > 1) {
        eps *= dev;
    }
    for(size_t i=0;i<size;i++) {
        if(fabs(a[i] - r[i]) > eps) {
            std::cout << "  Failed  at " << i << " " << a[i] << "!=" << r[i] << std::endl;
            return false;
        }

    }
    return true;
}

int main(int argc,char **argv)
{
    if(argc < 4) {
        std::cerr << "Usage ./test_net [-k] device net.json weights.json" << std::endl;
        std::cerr << "   -k - keep intermediate tensors, don't optimize memory reuse "<<std::endl;
        return 1;
    }
    bool keep = false;
    if(argv[1] == std::string("-k")) {
        argv++;
        argc--;
        keep=true;
    }
    dp::Context ctx(argv[1]);
    dp::Net net(ctx);
    if(keep)
        net.keep_intermediate_tensors(keep);
    std::cout << "Testing for " << ctx.name() << std::endl;
    net.mode(dp::CalculationsMode::train);
    net.load_from_json_file(argv[2]);
    net.setup();
    std::ifstream wt(argv[3]);
    dp::json::value w;
    if(!w.load(wt,true)) {
        std::cerr << "Failed to load weights " << argv[3] << std::endl;
        return 1;
    }
    dp::Context cpu_ctx;
    std::map<std::string,Tensor> diff_to_check;
    std::map<std::string,Tensor> param_to_check;
    auto e=ctx.make_queue();
    for(auto const &tdata : w.array()) {
        std::string name = tdata.get<std::string>("name");
        std::string type = tdata.get<std::string>("type");
        std::vector<float> value = tdata.get<std::vector<float> >("value");
        Tensor t;
        if(type == "param")
            t = net.param(name);
        else if(type == "data")
            t = net.tensor(name);
        else if(type == "data_diff")
            t = net.tensor_diff(name);
        else if(type == "param_diff")
            t = Tensor(cpu_ctx,net.param_diff(name).shape());
        else if(type == "param_updated")
            t = Tensor(cpu_ctx,net.param(name).shape());
        else {
            std::cerr << "Invaid type " << type << std::endl;
            return 1;
        }
        if(value.size() != t.shape().total_size()) {
            std::cerr << "Size mistmatch for " << name << std::endl;
            return 1;
        }
        memcpy(t.data<float>(),value.data(),value.size()*sizeof(float));
        t.to_device(e);
        if(type == "param_diff")
            diff_to_check[name] = t;
        else if(type == "param_updated")
            param_to_check[name] = t;
    }
    dp::solvers::Adam adam(ctx);
    adam.weight_decay = 0.05;
    adam.beta1 = 0.9;
    adam.beta2 = 0.95;
    adam.lr = 0.1;
    adam.eps = 1e-3;
    adam.init(net,e);
    adam.zero_grad(net,e);
    net.forward(e);
    net.backward(e);
    adam.apply(net,e);
    std::cout << "Checking Diffs" << std::endl;
    for(auto &p : diff_to_check) {
        auto name = p.first;
        Tensor res = net.param_diff(name);
        res.to_host(e);
        if(!compare_tensors(name,res,p.second))
            return 1;
    }
    std::cout << "Checking Param Updates" << std::endl;
    for(auto &p : param_to_check) {
        auto name = p.first;
        Tensor res = net.param(name);
        res.to_host(e);
        if(!compare_tensors(name,res,p.second))
            return 1;
    }
    return 0;
    
}
