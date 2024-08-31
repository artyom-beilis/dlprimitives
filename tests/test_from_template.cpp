///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/operator.hpp>
#include <dlprim/json.hpp>
#include <fstream>
#include <iostream>
#include "test.hpp"

namespace dp = dlprim;


std::vector<dp::TensorSpecs> tensor_specs_from_json(dp::json::value const &v)
{
    std::vector<dp::TensorSpecs> r;
    dp::json::array const &ts = v.array();
    for(dp::json::value const &spec : ts) {
        std::vector<int> vsp = spec.get<std::vector<int> >("shape");
        dp::Shape sp=dp::Shape::from_range(vsp.begin(),vsp.end());
        dp::DataType dtype = dp::string_to_data_type(spec.get<std::string>("dtype","float"));
        bool is_trainable = spec.get<bool>("trainable",true);
        r.push_back(dp::TensorSpecs(sp,dtype,is_trainable));
    }
    return r;
}

std::vector<int> get_requires_grad(dp::json::value const &v)
{
    std::vector<int> rg;
    for(auto &val : v.array()) {
        rg.push_back(int(val.get("requires_grad",true)));
    }
    return rg;
}

std::vector<dp::Shape> tensor_shapes_from_json(dp::json::value const &v)
{
    std::vector<dp::Shape> r;
    dp::json::array const &sps = v.array();
    for(dp::json::value const &jsp : sps) {
        std::vector<int> vsp = jsp.get_value<std::vector<int> >();
        dp::Shape sp=dp::Shape::from_range(vsp.begin(),vsp.end());
        r.push_back(sp);
    }
    return r;
}

std::vector<dp::Tensor> make_tensors(dp::Context &ctx,
                                     std::vector<dp::TensorSpecs> const &specs)
{
    std::vector<dp::Tensor> r;
    for(size_t i=0;i<specs.size();i++) {
        r.push_back(dp::Tensor(ctx,specs[i].shape(),specs[i].dtype()));
    }
    return r;
}


std::vector<dp::Tensor> make_tensors(dp::Context &ctx,
                                     std::vector<dp::Shape> const &shapes,
                                     std::vector<dp::TensorSpecs> const &specs)
{
    std::vector<dp::Tensor> r;
    for(size_t i=0;i<shapes.size();i++) {
        r.push_back(dp::Tensor(ctx,shapes[i],specs[i].dtype()));
    }
    return r;
}

void copy_tensors(std::vector<dp::Tensor> &tensors,dp::json::value const &v)
{
    dp::json::array const &vals = v.array();
    for(size_t i=0;i<tensors.size();i++) {
        if(vals.at(i).type() == dp::json::is_boolean && vals[i].get_value<bool>()==false) {
            continue;
        }
        std::vector<float> const &values = vals.at(i).get_value<std::vector<float> >();
        if(values.size() !=tensors[i].shape().total_size()) {
            std::cerr << "Got vector of size " << values.size() << " for shape " << tensors[i].shape() << std::endl;
        }
        TESTEQ(values.size(),tensors[i].shape().total_size());
        if(tensors[i].dtype() == dp::float_data) {
            float *p=tensors[i].data<float>();
            memcpy(p,values.data(),sizeof(float)*tensors[i].shape().total_size());
        }
        else {
            TEST(!"Unsuported data type");
        }
    }
}

void copy_tensors(std::vector<dp::Tensor> &out,std::vector<dp::Tensor> &inp,dp::Context &ctx)
{
    TESTEQ(out.size(),inp.size());
    cl::CommandQueue q = ctx.make_queue();
        
    for(size_t i=0;i<out.size();i++) {
        TESTEQ(out[i].shape(),inp[i].shape());
        TESTEQ(out[i].dtype(),inp[i].dtype());
        size_t len = dp::size_of_data_type(out[i].dtype()) * out[i].shape().total_size();
        memcpy(out[i].host_data(),inp[i].host_data(),len);
        out[i].to_device(q,true);
    }
}


void compare_tensors(std::vector<dp::Tensor> &actual,std::vector<dp::Tensor> &reference,
                float eps,float factor=1.0f,char const *name="",std::vector<int> const &grad=std::vector<int>())
{
    bool failed = false;
    for(size_t i=0;i < actual.size();i++) {
        if(i < grad.size() && !grad[i])
            continue;
        int fail_counts=0;
        if(actual[i].dtype() == dp::float_data) {
            float *a=actual[i].data<float>();
            float *r=reference[i].data<float>();
            int total =  actual[i].shape().total_size();
            if(total == 0)
                continue;
            float sum=0,sum2=0;
            for(int j=0;j<total;j++) {
                sum +=r[j];
                sum2+=r[j]*r[j];
            }
            sum /= total;
            sum2 /= total;
            float variance = sum2 - sum*sum;
            float dev = std::sqrt(variance);
            //float dev = std::sqrt(sum2);
            if(dev > 1) {
                eps *= dev;
            }
            else if(total == 1 && std::fabs(sum) > 1) {
                eps *= std::fabs(sum);
            }
            for(int j=0;j<total && fail_counts < 10;j++) {
                if(fabs(a[j] - factor * r[j]) > eps) {
                    std::cerr << "Comparison failed for tensor " << name 
                        << ":" << i << " at " << j << " expecting " << r[j]*factor << "=" << r[j] << "*" << factor << " got "
                         << a[j] << " for esp="<<eps << std::endl;
                    fail_counts ++;
                    failed=true;
                }
            }
        }
        else {
            TEST(!"Unsuported data type");
        }
    }
    if(failed) {
        throw std::runtime_error("Computations Failed");
    }
}

void fill_random(dp::Tensor &t)
{
    TEST(t.dtype() == dp::float_data);
    float *p = t.data<float>();
    for(size_t i=0;i<t.shape().total_size();i++) {
        p[i]=rand()*2.0f / RAND_MAX - 1.0f;
    }
}

void fill_small_int(dp::Tensor &t,int range,float factor = 1.0)
{
    TEST(t.dtype() == dp::float_data);
    float *p = t.data<float>();
    for(size_t i=0;i<t.shape().total_size();i++) {
        int v = int(double(rand()) / RAND_MAX * range) - range / 2;
        p[i]= v * factor;
    }
}


void initialize_tensors(std::vector<dp::Tensor> &tensors,std::string const &op)
{
    for(auto &t:tensors) {
        if(op == "uniform")
            fill_random(t);
        else if(op == "small_int")
            fill_small_int(t,5,1.0f);
        else if(op == "small_frac")
            fill_small_int(t,5,0.5f);
        else
            throw std::runtime_error("Unknown methods " + op);
    }
}

std::vector<dp::TensorAndGradient> join_grad(std::vector<dp::Tensor> &data,std::vector<dp::Tensor> &diff,float accum = 0)
{
    std::vector<dp::TensorAndGradient> joined(data.size());
    for(size_t i=0;i<data.size();i++) {
        joined[i].data = data[i];
        joined[i].diff = diff[i];
        joined[i].requires_gradient = true;
        joined[i].accumulate_gradient = accum;
    }
    return joined;
}


std::vector<dp::TensorAndGradient> join_grad(std::vector<dp::Tensor> &data,std::vector<dp::Tensor> &diff,std::vector<int> &rg,float accum = 0)
{
    std::vector<dp::TensorAndGradient> joined(data.size());
    for(size_t i=0;i<data.size();i++) {
        joined[i].data = data[i];
        joined[i].diff = diff[i];
        joined[i].requires_gradient = rg[i];
        joined[i].accumulate_gradient = accum;
    }
    return joined;
}

int main(int argc,char **argv)
{
    if(argc != 3) {
        std::cerr << "Usage test_from_template context_id test.json" << std::endl;
        return 1;
    }
    try {
        dp::Context cpu_ctx;
        dp::ExecutionContext cpu_e;

        dp::Context ctx(argv[1]);
        dp::ExecutionContext e;
        if(!ctx.is_cpu_context()) {
            cl::CommandQueue q = ctx.make_queue();
            e = dp::ExecutionContext(q);
        }
        
        std::ifstream f(argv[2]);
        if(!f) {
            std::cerr << "Failed to open " << argv[2] << std::endl;
            return 1;
        }
        dp::json::value config;
        int line = -1;
        if(!config.load(f,true,&line)) {
            std::cerr << "Failed to load json template " << argv[2] << " syntax error at " << line << std::endl;
            return 1;
        }
        
        std::string op_name = config.get<std::string>("operator");
        std::cout << "Running tests for operator " << op_name << " on " << ctx.name() << std::endl;
        dp::json::array &tests = config["tests"].array();
        for(size_t i=0;i<tests.size();i++) {
            std::cout << "- Running test case " << i << " for options " << tests[i]["options"] << std::endl;
            bool train = tests[i].get("train",false);
            std::string rnd = tests[i].get("init","uniform");
            std::unique_ptr<dp::Operator> op =     dp::create_by_name(ctx,    op_name,tests[i]["options"]);
            std::unique_ptr<dp::Operator> ref_op = dp::create_by_name(cpu_ctx,op_name,tests[i]["options"]);

            std::vector<dp::TensorSpecs> input_specs = tensor_specs_from_json(tests[i]["setup_tensors"]);
            std::vector<int> grad = get_requires_grad(tests[i]["setup_tensors"]);
            std::vector<dp::TensorSpecs> output_specs = tensor_specs_from_json(tests[i]["output_tensors"]);
            int ws = tests[i].get("workspace",-1);
            std::vector<dp::TensorSpecs> res_specs,res_param_specs,ref_specs,ref_param_specs;
            size_t res_ws;
            dp::Tensor res_ws_tensor;
            if(train) {
                op->mode(dp::CalculationsMode::train);
                ref_op->mode(dp::CalculationsMode::train);
            }
            op->setup(input_specs,res_specs,res_param_specs,res_ws);
            if(res_specs != output_specs) {
                std::cerr << "Res"<<std::endl;
                for(auto x:res_specs)
                    std::cerr <<x<<std::endl;
                std::cerr << "out"<<std::endl;
                for(auto x:output_specs)
                    std::cerr <<x<<std::endl;
            }
            TEST(res_specs == output_specs);
            TEST(ws == -1 || res_ws == size_t(ws));
            if(res_ws > 0) {
                res_ws_tensor = dp::Tensor(ctx,dp::Shape(res_ws),dp::uint8_data);
            }
            size_t ref_ws;
            ref_op->setup(input_specs,ref_specs,ref_param_specs,ref_ws);
            dp::Tensor ref_ws_tensor;
            if(ref_ws) {
                ref_ws_tensor = dp::Tensor(cpu_ctx,dp::Shape(ref_ws),dp::uint8_data);
            }
            TEST(res_specs == output_specs);
            std::vector<dp::Tensor> params,ref_params;
            std::vector<dp::TensorSpecs> param_specs;
            if(!tests[i].find("param_specs").is_undefined()) {
                param_specs = tensor_specs_from_json(tests[i]["param_specs"]);
                if(res_param_specs != param_specs) {
                    for(size_t i=0;i<res_param_specs.size();i++)
                        std::cerr << res_param_specs.at(i) << " " << param_specs.at(i)  << std::endl;
                }
                TEST(res_param_specs == param_specs);
                params = make_tensors(ctx,param_specs);
                ref_params = make_tensors(cpu_ctx,param_specs);
                if(!tests[i].get("random_params",false)) {
                    copy_tensors(ref_params,tests[i]["param_tensors"]);
                }
                else {
                   initialize_tensors(ref_params,rnd);
                }
                copy_tensors(params,ref_params,ctx);
            }
            dp::json::array const &cases = tests[i]["cases"].array();
            for(size_t i=0;i<cases.size();i++) {
                bool use_cpu_reference = cases[i].get("use_cpu_reference",false);
                bool case_train = !cases[i].get("test_mode",!train);
                if(case_train) {
                    op->mode(dp::CalculationsMode::train);
                    ref_op->mode(dp::CalculationsMode::train);
                }
                else {
                    op->mode(dp::CalculationsMode::predict);
                    ref_op->mode(dp::CalculationsMode::predict);
                }

                std::cout << "-- test for shape " << cases[i]["in_shapes"] << " fwd" << std::flush;
                std::vector<dp::Shape> in_shapes = tensor_shapes_from_json(cases[i]["in_shapes"]);
                std::vector<dp::Shape> out_shapes = tensor_shapes_from_json(cases[i]["out_shapes"]);
                std::vector<dp::Shape> res_shape;
                size_t ws_size = 0;
                op->reshape(in_shapes,res_shape,ws_size);
                if(ws_size > res_ws) {
                    res_ws = ws_size;
                    res_ws_tensor = dp::Tensor(ctx,dp::Shape(res_ws),dp::uint8_data);
                }
                if(out_shapes != res_shape) {
                    std::cerr << "Res" <<std::endl;
                    for(auto s:res_shape)
                        std::cerr << s << std::endl;
                    std::cerr << "Out" <<std::endl;
                    for(auto s:out_shapes)
                        std::cerr << s << std::endl;
                }
                TEST(out_shapes == res_shape);
                ws_size = 0;
                ref_op->reshape(in_shapes,res_shape,ws_size);
                if(ws_size > ref_ws) {
                    ref_ws = ws_size;
                    ref_ws_tensor = dp::Tensor(cpu_ctx,dp::Shape(ref_ws),dp::uint8_data);
                }
                TEST(out_shapes == res_shape);
                std::vector<dp::Tensor> in_tensors = make_tensors(ctx,in_shapes,input_specs);
                std::vector<dp::Tensor> out_tensors = make_tensors(ctx,out_shapes,output_specs);
                std::vector<dp::Tensor> ref_tensors = make_tensors(cpu_ctx,out_shapes,output_specs);
                if(!use_cpu_reference) {
                    copy_tensors(in_tensors,cases[i]["in_tensors"]);
                    copy_tensors(ref_tensors,cases[i]["out_tensors"]);
                }
                else {
                    initialize_tensors(in_tensors,rnd);
                    ref_op->forward(in_tensors,ref_tensors,ref_params,ref_ws_tensor,cpu_e);
                }
                if(ctx.is_opencl_context()) {
                    for(dp::Tensor &tensor : in_tensors)
                        tensor.to_device(e,false);
                }
                op->forward(in_tensors,out_tensors,params,res_ws_tensor,e);
                if(ctx.is_opencl_context()) {
                    for(dp::Tensor &tensor : out_tensors)
                        tensor.to_host(e,false);
                    if(ctx.is_opencl_context())    
                        e.queue().finish();
                }
                double eps = cases[i].get<double>("eps",1e-5);
                compare_tensors(out_tensors,ref_tensors,eps);
                if(train && (use_cpu_reference || !cases[i].find("out_diffs").is_undefined())) {
                    std::cout <<",bwd" << std::flush;
                    std::vector<dp::Tensor> out_diffs   = make_tensors(ctx,out_shapes,output_specs);
                    std::vector<dp::Tensor> ref_diffs   = make_tensors(cpu_ctx,out_shapes,output_specs);
                    std::vector<dp::Tensor> in_diffs    = make_tensors(ctx,in_shapes,input_specs);
                    std::vector<dp::Tensor> in_ref_diffs    = make_tensors(cpu_ctx,in_shapes,input_specs);
                    std::vector<dp::Tensor> param_diffs = make_tensors(ctx,param_specs);
                    std::vector<dp::Tensor> param_ref_diffs = make_tensors(cpu_ctx,param_specs);
                    if(!use_cpu_reference) {
                        copy_tensors(out_diffs,cases[i]["out_diffs"]);
                        copy_tensors(in_ref_diffs,cases[i]["in_diffs"]);
                        if(!params.empty()) {
                            copy_tensors(param_ref_diffs,cases[i]["params_diffs"]);
                        }
                    }
                    else {
                        initialize_tensors(out_diffs,rnd);
                        auto a = join_grad(in_tensors,in_ref_diffs,grad,0);
                        auto b = join_grad(ref_tensors,out_diffs,0);
                        auto c = join_grad(ref_params,param_ref_diffs,0);
                        ref_op->backward(a,b,c,ref_ws_tensor,cpu_e);
                    }
                    int accums = cases[i].get<bool>("double_check",true) ? 2 : 1;
                    for(int accum = 0;accum<accums;accum++) {
                        if(ctx.is_opencl_context()) {
                            for(dp::Tensor &tensor : out_diffs)
                                tensor.to_device(e,false);
                        }
                        {
                            auto a = join_grad(in_tensors,in_diffs,grad,accum * 0.5f);
                            auto b = join_grad(out_tensors,out_diffs);
                            auto c = join_grad(params,param_diffs,accum * 0.5f);
                            op->backward(a,b,c,res_ws_tensor,e);
                        }
                        if(ctx.is_opencl_context()) {

                            for(dp::Tensor &tensor : in_diffs)
                                tensor.to_host(e,false);
                            for(dp::Tensor &tensor : param_diffs)
                                tensor.to_host(e,false);
                            if(ctx.is_opencl_context())    
                                e.queue().finish();
                        }
                        std::vector<int> params_grad,params_nograd;
                        for(auto p : param_specs) {
                            params_grad.push_back(p.is_trainable());
                            params_nograd.push_back(!p.is_trainable());
                        }
                        compare_tensors(in_diffs,in_ref_diffs,eps,1.0 + accum * 0.5f,"data",grad);
                        compare_tensors(param_diffs,param_ref_diffs,eps,1.0 + accum * 0.5f,"filter",params_grad);
                        if(!cases[i].find("parameters_check").is_undefined()) {
                            for(dp::Tensor &tensor : params)
                                tensor.to_host(e);
                            std::vector<dp::Tensor> parameters_check_ref = make_tensors(cpu_ctx,param_specs);
                            copy_tensors(parameters_check_ref,cases[i]["parameters_check"]);
                            compare_tensors(params,parameters_check_ref,eps,1.0,"params",params_nograd);
                        }
                    }
                }
                std::cout << std::endl;
            }
        }

        std::cout << "Testing " << argv[2] << " completed sucesefully" << std::endl;
        
        return 0;
    }
    catch(cl::Error const &e) {
        std::cerr << "\n\nFAILED: " << e.what() << " " << e.err()<< std::endl;
        return 1;
    }
    catch(std::exception const &e) {
        std::cerr << "\n\nFAILED: " << e.what() << std::endl;
        return 1;
    }
    
}
