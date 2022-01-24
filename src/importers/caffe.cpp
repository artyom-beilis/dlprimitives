#include <dlprim/caffe.hpp>
#include <dlprim/json.hpp>

#include "caffe_dlprim.pb.h"

#include <fstream>
#include <set>
#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/text_format.h>

namespace dlprim {
    struct CaffeModel::Data {
        caffe::NetParameter model;
        caffe::NetParameter params;
        json::value net;
        std::map<std::string,Tensor> parameters;
        std::map<std::string,int> edges; // number of dims per edge
        int counter = 0;
    };

    CaffeModel::CaffeModel() : d(new CaffeModel::Data())
    {
        d->net["inputs"] = json::array();
        d->net["outputs"] = json::array();
        d->net["operators"] = json::array();
    }
    CaffeModel::~CaffeModel() 
    {
    }
    void CaffeModel::get_outputs()
    {
        std::set<std::string> outputs;
        for(int i=0;i<d->model.layer_size();i++) {
            caffe::LayerParameter const &layer = d->model.layer(i);
            for(auto const &name : layer.bottom())
                outputs.erase(name);
            for(auto const &name : layer.top())
                outputs.insert(name);
        }
        d->net["outputs"] = std::vector<std::string>(outputs.begin(),outputs.end());
    }
    void CaffeModel::load(std::string const &proto_file_name,std::string const &caffe_model)
    {
        load_text_proto(proto_file_name);
        load_binary_proto(caffe_model);
        load_parameters();
        get_outputs();
        if(d->model.input_size() > 0) {
            add_deprecated_input();
        }
        add_layers();
    }
    void CaffeModel::add_deprecated_input()
    {
        if(d->model.input_shape_size()) {
            DLPRIM_CHECK(d->model.input_size() == d->model.input_shape_size());
            for(int i=0;i<d->model.input_size();i++) {
                d->net["inputs"][i]["name"] = d->model.input(i);
                std::vector<int> shape(d->model.input_shape(i).dim().begin(),d->model.input_shape(i).dim().end());
                d->net["inputs"][i]["shape"]  = shape;
                d->edges[d->model.input(i)] = shape.size();
            }
        }
        else if(d->model.input_size()!=0) {
            DLPRIM_CHECK(d->model.input_size()==1);
            d->net["inputs"][0]["name"] = d->model.input(0);
            std::vector<int> shape(d->model.input_dim().begin(),d->model.input_dim().end());
            d->net["inputs"][0]["shape"]  = shape;
            d->edges[d->model.input(0)] = shape.size();
        }
    }
    void CaffeModel::check_inputs(caffe::LayerParameter const &layer,int inputs_min,int inputs_max)
    {
        if(inputs_max == -1)
            inputs_max = inputs_min;
        int size = layer.bottom_size();
        if(!(inputs_min <= size && size <= inputs_max))
            throw ValidationError("Invalid number of inputs for layer " + layer.type() + "/" + layer.name());
        for(int index = 0;index<size;index++) {
            if(d->edges.find(layer.bottom(index))==d->edges.end()) {
                throw ValidationError("Can't find input " + layer.bottom(index) + " for operator " + layer.type() + "/" + layer.name());
            }
        }
    }
    std::string CaffeModel::param_name(caffe::LayerParameter const &layer,int index,bool check)
    {
        std::string name;
        if(index < layer.param_size() && !layer.param(index).name().empty())
            name = layer.param(index).name();
        else
            name = param_name(layer.name(),index);
        if(check) {
            if(d->parameters.find(name) == d->parameters.end())
                throw ValidationError("No parameter " + name + " for layer " + layer.name());
        }
        return name;
    }
    std::string CaffeModel::param_name(std::string const &layer,int index)
    {
        return layer + "[" + std::to_string(index) + "]";
    }
    void CaffeModel::load_parameters()
    {
        for(caffe::LayerParameter const &layer : d->params.layer()) {
            std::string name = layer.name();
            for(int i=0;i<layer.blobs_size();i++) {
                d->parameters[param_name(name,i)] = to_tensor(layer.blobs(i));
            }
        }
    }
    Tensor CaffeModel::to_tensor(caffe::BlobProto const &blob)
    {
        auto const &dims = blob.shape().dim();
        Shape sp=Shape::from_range(dims.begin(),dims.end());
        Context ctx;
        Tensor res(ctx,sp);
        DLPRIM_CHECK(sp.total_size() == size_t(blob.data_size()));
        float *ptr = res.data<float>();
        for(float v : blob.data()) {
            *ptr ++ = v;
        }
        return res;
    }
    void CaffeModel::check_outputs(caffe::LayerParameter const &layer,int minv,int maxv,bool allow_inplace)
    {
        if(maxv == -1)
            maxv = minv;
        DLPRIM_CHECK(minv <= layer.top_size() && layer.top_size() <= maxv);
        for(int i=0;i<layer.top_size();i++) {
            std::string const &name = layer.top(i);
            if(allow_inplace && i<layer.bottom_size() && name == layer.bottom(i))
                continue;
            DLPRIM_CHECK(d->edges.find(name) == d->edges.end());
        }
    }
    
    void CaffeModel::get_conv_params(caffe::ConvolutionParameter const &lp,json::value &opt)
    {
        if(lp.has_kernel_w() || lp.has_kernel_h()) {
            opt["kernel"][0] = lp.kernel_h();
            opt["kernel"][1] = lp.kernel_w();
        }
        else if(lp.kernel_size_size() > 1) {
            for(int i=0;i<lp.kernel_size_size();i++)
                opt["kernel"][i] = lp.kernel_size(i);
        }
        else if(lp.kernel_size_size() == 1) {
            opt["kernel"] = lp.kernel_size(0);
        }
        else {
            throw ValidationError("Invalid kernel size specified");
        }
        if(lp.has_pad_w() || lp.has_pad_h()) {
            opt["pad"][0] = lp.pad_h();
            opt["pad"][1] = lp.pad_w();
        }
        else if(lp.pad_size() > 1) {
            for(int i=0;i<lp.pad_size();i++)
                opt["pad"][i] = lp.pad(i);
        }
        else if(lp.pad_size() == 1) {
            opt["pad"] = lp.pad(0);
        }
        if(lp.has_stride_w() || lp.has_stride_h()) {
            opt["stride"][0] = lp.stride_h();
            opt["stride"][1] = lp.stride_w();
        }
        else if(lp.stride_size() > 1) {
            for(int i=0;i<lp.stride_size();i++)
                opt["stride"][i] = lp.stride(i);
        }
        else if(lp.stride_size() == 1) {
            opt["stride"] = lp.stride(0);
        }
        if(lp.dilation_size() > 1) {
            for(int i=0;i<lp.dilation_size();i++)
                opt["dilate"][i] = lp.dilation(i);
        }
        else if(lp.dilation_size() == 1) {
            opt["dilate"] = lp.dilation(0);
        }
    }

    void CaffeModel::add_operator(caffe::LayerParameter const &layer,json::value &v,int N,bool add_outputs)
    {
        d->net["operators"].array().push_back(std::move(v));
        if(add_outputs) {
            for(std::string const &output : layer.top()) {
                d->edges[output] = N;
            }
        }
    }

    void CaffeModel::add_conv(caffe::LayerParameter const &layer)
    {
        check_inputs(layer,1);
        check_outputs(layer,1);
        json::value v;
        v["name"] = layer.name();
        v["type"] = layer.type() == "Deconvolution"  ? "TransposedConvolution2D" : "Convolution2D";
        v["inputs"][0] = layer.bottom(0);
        v["outputs"][0] = layer.top(0);
        v["params"][0] = param_name(layer,0);
        caffe::ConvolutionParameter const &lp = layer.convolution_param();
        bool bias = lp.bias_term();
        if(bias)
            v["params"][1] = param_name(layer,1);
        json::value &opt=v["options"];
        int groups = lp.group();
        DLPRIM_CHECK(lp.axis() == 1);
        opt["bias"] = bias;
        opt["groups"] = groups;
        get_conv_params(lp,opt);
        opt["channels_out"]=lp.num_output();

        add_operator(layer,v,4);

    }
    std::string CaffeModel::remove_inplace(std::string const &name)
    {
        std::string new_name = name + "_DLPRIM_remove_inline";
        int N = d->edges[name];
        d->edges.erase(name);
        d->edges[new_name] = N;
        json::array &operators = d->net["operators"].array();
        for(json::value &v : operators) {
            for(json::value &out : v["outputs"].array()) {
                if(out.str() == name) {
                    out = new_name;
                    return new_name;
                }
            }
        }
        for(json::value &inp : d->net["inputs"].array()) {
            if(inp["name"].str() == name) {
                inp["name"] = new_name;
                return new_name;
            }
        }
        throw ValidationError("Can't find input " + name);
    }
    
    json::value *CaffeModel::get_input_generator(std::string const &name)
    {
        json::array &operators = d->net["operators"].array();
        for(json::value &v : operators) {
            json::array &outputs = v["outputs"].array();
            if(outputs.empty())
                continue;
            if(outputs[0].str() == name)
                return &v;
        }
        return nullptr;
    }

    bool CaffeModel::is_inplace(caffe::LayerParameter const &layer)
    {
        return layer.bottom_size() == 1 && layer.top_size() == 1 && layer.bottom(0) == layer.top(0);
    }

    void CaffeModel::add_bn(caffe::LayerParameter const &layer)
    {
        check_inputs(layer,1);
        DLPRIM_CHECK(layer.top_size() == 1);
        std::string input;
        if(is_inplace(layer))
            input = remove_inplace(layer.bottom(0));
        else 
            input = layer.bottom(0);
        json::value op;
        op["name"] = layer.name();
        op["inputs"][0] = input;
        op["outputs"][0] = layer.top(0);
        op["type"] = "BatchNorm";
        op["options"]["eps"] = layer.batch_norm_param().eps();
        op["options"]["affine"] = false;
        std::string mean = param_name(layer,0);
        std::string var = param_name(layer,1);
        std::string scale = param_name(layer,2);
        auto p=d->parameters.find(scale);
        if(p!=d->parameters.end() && p->second.shape() == Shape(1)) {
            Tensor src_mean = get_parameter(mean);
            Tensor src_var = get_parameter(var);
            Context ctx;
            Tensor mean_fixed(ctx,src_mean.shape());
            Tensor var_fixed(ctx,src_mean.shape());
            float factor = 1.0f / p->second.data<float>()[0];
            size_t N = mean_fixed.shape().total_size();
            float *mf=mean_fixed.data<float>();
            float *vf=var_fixed.data<float>();
            float *ms=src_mean.data<float>();
            float *vs=src_var.data<float>();
            for(size_t i=0;i<N;i++) {
                mf[i] = ms[i]*factor;
                vf[i] = vs[i]*factor;
            }
            mean += "_DLP_Fix";
            var += "_DLP_Fix";
            d->parameters[mean] = mean_fixed;
            d->parameters[var] = var_fixed;
        }
        op["params"][0] = mean;
        op["params"][1] = var;
        add_operator(layer,op,d->edges[input]);
    }
    
    bool CaffeModel::is_mergable(caffe::LayerParameter const &layer)
    {
        std::string edge = layer.bottom(0);
        int counter = 0;
        for(auto const &l : d->model.layer()) {
            for(int i=0;i<l.bottom_size();i++) {
                if(l.bottom(i)==edge && !(i<l.top_size() && l.top(i) == l.bottom(i)))
                    counter++;
            }
        }
        return counter == 1;
    }
    void CaffeModel::add_scale(caffe::LayerParameter const &layer)
    {
        check_inputs(layer,1);
        json::value *inp = get_input_generator(layer.bottom(0));
        bool inplace = false,mergable=false;
        bool bias = layer.scale_param().bias_term();
        DLPRIM_CHECK(layer.scale_param().axis() == 1);
        DLPRIM_CHECK(layer.scale_param().num_axes() == 1);
        if( inp 
            && inp->get<std::string>("type") == "BatchNorm" 
            && inp->get<bool>("options.affine")==false
            && ((inplace=is_inplace(layer)) || (mergable=is_mergable(layer)))
            && bias)
        {
            json::value &op = *inp;
            op["params"][2] = param_name(layer,0);
            op["params"][3] = param_name(layer,1);
            op["options"]["affine"] = true;
            if(mergable) {
                op["outputs"][0] = layer.top(0);
                int N = d->edges[layer.bottom(0)];
                d->edges.erase(layer.bottom(0));
                d->edges[layer.top(0)] = N;
            }
            return;
        }
        std::string input_name = non_inplace_input(layer); 
        std::string output_name = layer.top(0);
        std::string p1 = param_name(layer,0);
        std::string p2;
        Tensor W = get_parameter(p1);
        Tensor B;
        if(bias) {
            p2 = param_name(layer,1);
            B = get_parameter(p2);
        }
        if(d->edges[input_name] == 4) {
            json::value op;
            size_t N = W.shape().total_size();
            DLPRIM_CHECK(N > 0);
            p1 += "_DLPrim_Reshape";
            W = W.alias();
            W.reshape(Shape(N,1,1,1));
            d->parameters[p1] = W;
            op["params"][0] = p1;
            if(bias) {
                DLPRIM_CHECK(B.shape().total_size() == N);
                if(B.shape() != Shape(N)) {
                    p2 += "_DLPrim_Reshape";
                    B = B.alias();
                    B.reshape(Shape(N));
                    d->parameters[p2] = B;
                }
                op["params"][1] = p2;
            }
            op["name"] = layer.name();
            op["type"] = "Convolution2D";
            op["inputs"][0] = input_name;
            op["outputs"][0] = layer.top(0);
            json::value &opt = op["options"];
            opt["bias"] = bias;
            opt["groups"] = N;
            opt["channels_out"] = N;
            opt["channels_in"] = N;
            opt["kernel"] = 1;
            add_operator(layer,op,4);
        }
        else {
            DLPRIM_CHECK(W.shape().size() == 1);
            if(bias) {
                DLPRIM_CHECK(W.shape() == B.shape());
            }
            int input_rank = d->edges[input_name];
            int N = W.shape()[0];
            std::vector<int> dims={N};
            for(int i=2;i<input_rank;i++)
                dims.push_back(1);
            Shape new_shape=Shape::from_range(dims.begin(),dims.end());
            if(new_shape != W.shape()) {
                W = W.alias();
                W.reshape(new_shape);
                p1 += "_DLPrim_Reshape";
                d->parameters[p1] = W;
                if(bias) {
                    B = B.alias();
                    p2 += "_DLPrim_Reshape";
                    B.reshape(new_shape);
                }
            }
            json::value op;
            op["name"] = p1;
            op["type"] = "Parameter";
            op["inputs"] = json::array();
            op["outputs"][0] = p1;
            op["params"][0] = p1;
            op["options"]["shape"] = dims;
            json::value tmp_op = op;
            add_operator(layer,tmp_op,0,false);
            d->edges[p1] = dims.size();

            std::string output_scal = bias ? layer.top(0) + "_dlprim_scal" : layer.top(0);
            json::value scal;
            scal["name"] = layer.name();
            scal["type"] = "Elementwise";
            scal["inputs"][0] = input_name;
            scal["inputs"][1] = p1;
            scal["outputs"][0] = output_scal;
            scal["options"]["operation"] = "prod";
            json::value tmp_scal = scal;
            add_operator(layer,tmp_scal,0,false);
            d->edges[output_scal] = input_rank;

            if(bias) {
                op["name"] = p2;
                op["outputs"][0] = p2;
                op["params"][0] = p2;
                add_operator(layer,op,0,false);
                d->edges[p2] = dims.size();
                scal["name"] = layer.name() + "_dlprim_bias";
                scal["inputs"][0] = output_scal;
                scal["inputs"][1] = p2;
                scal["outputs"][0] = layer.top(0);
                scal["options"]["operation"] = "sum";
                add_operator(layer,scal,input_rank);
            }
        }
    }

    void CaffeModel::add_standard_activation(caffe::LayerParameter const &layer,std::string const &name,bool validate_inputs)
    {
        if(validate_inputs) {
            check_inputs(layer,1);
            if(layer.top_size()!=1 || layer.top(0) != layer.bottom(0)) {
                check_outputs(layer,1);
            }
        }
        std::string type;
        json::array &ops = d->net["operators"].array();
        if( !ops.empty()
            && is_mergable(layer)
            && ((type=ops.back()["type"].str()) == "Convolution2D" || type == "InnerProduct" || type == "Elementwise")
            && ops.back()["options"].get("activation","identity") == "identity"
            && ops.back()["outputs"][0] == layer.bottom(0))
        {
            int N = d->edges[layer.bottom(0)];
            d->edges.erase(layer.bottom(0));
            d->edges[layer.top(0)] = N;
            ops.back()["outputs"][0] = layer.top(0);
            ops.back()["options"]["activation"] = name;
        }
        else {
            json::value op;
            op["name" ] = layer.name();
            op["type" ] = "Activation";
            op["inputs"][0] = layer.bottom(0);
            op["outputs"][0] = layer.top(0);
            op["options"]["activation"] = name;
            add_operator(layer,op,d->edges[layer.bottom(0)]);
        }
    }
    
    void CaffeModel::add_pool2d(caffe::LayerParameter const &layer)
    {
        check_inputs(layer,1);
        check_outputs(layer,1);
        caffe::PoolingParameter const &par = layer.pooling_param();
        json::value op;
        op["name"] = layer.name();
        op["inputs"][0] = layer.bottom(0);
        op["outputs"][0] = layer.top(0);
        json::value &opt = op["options"];
        std::string mode;
        if(par.pool() == par.MAX)
            mode = "max";
        else if(par.pool() == par.AVE)
            mode = "avg";
        else
            throw ValidationError("Stochasting pooling is not supported");
        opt["mode"] = mode;

        if(par.global_pooling()) {
            op["type"] = "GlobalPooling";
        }
        else {
            op["type"] = "Pooling2D";
            if(par.has_pad_h() || par.has_pad_w())  {
                opt["pad"][0] = par.pad_h();
                opt["pad"][1] = par.pad_w();
            }
            else {
                opt["pad"] = par.pad();
            }
            if(par.has_kernel_h() || par.has_kernel_w()) {
                opt["kernel"][0] = par.kernel_h();
                opt["kernel"][1] = par.kernel_w();
            }
            else{
                opt["kernel"] = par.kernel_size();
            }
            if(par.has_stride_h() || par.has_stride_w())  {
                opt["stride"][0] = par.stride_h();
                opt["stride"][1] = par.stride_w();
            }
            else {
                opt["stride"] = par.stride();
            }
            opt["count_include_pad"] = false;
            opt["ceil_mode"] = true;
        }
        add_operator(layer,op,4);
    }
    void CaffeModel::add_elementwise(caffe::LayerParameter const &layer)
    {
        check_inputs(layer,2,std::numeric_limits<int>::max());
        check_outputs(layer,1);
        auto const &par = layer.eltwise_param();
        std::string last_output;
        for(int i=1;i<layer.bottom_size();i++) {
            std::string left = last_output.empty() ? layer.bottom(i-1) : last_output;
            std::string right = layer.bottom(i);
            std::string result = (i + 1 == layer.bottom_size()) ? layer.top(0) : layer.top(0) + "_eltwize_intermed_" + std::to_string(i);
            last_output = result;
            float coeff1 = 1.0, coeff2 = 1.0;
            if(last_output.empty() && i-1 < par.coeff_size())
                coeff1 = par.coeff(i-1);
            if(i < par.coeff_size())
                coeff2 = par.coeff(i);
            std::string name = (i + 1 == layer.bottom_size()) ? layer.name() : layer.name() + "_eltwise_name_" + std::to_string(i);
            json::value op;
            op["name"] = name;
            op["type"] = "Elementwise";
            op["inputs"][0] = left;
            op["inputs"][1] = right;
            op["outputs"][0] = result;
            json::value &opt = op["options"];
            std::string op_name;
            if(par.operation() == par.MAX)
                op_name = "max";
            else if(par.operation() == par.SUM)
                op_name = "sum";
            else if(par.operation() == par.PROD)
                op_name = "prod";
            else
                throw ValidationError("Operation not supported elementwise");
            opt["operation"] = op_name;
            opt["coef1"] = coeff1;
            opt["coef2"] = coeff2;
            add_operator(layer,op,0,false);
            d->edges[result] = std::max(d->edges[left],d->edges[right]);
        }
    }
    
    void CaffeModel::add_softmax(caffe::LayerParameter const &layer)
    {
        check_inputs(layer,1);
        check_outputs(layer,1);
        DLPRIM_CHECK(layer.softmax_param().axis() == 1 || layer.softmax_param().axis() == -1);
        json::value op;
        op["name"] = layer.name();
        op["type"] = "Softmax";
        op["inputs"][0] = layer.bottom(0);
        op["outputs"][0] = layer.top(0);
        add_operator(layer,op,2);

    }
    
    void CaffeModel::add_ip(caffe::LayerParameter const &layer)
    {
        check_inputs(layer,1);
        check_outputs(layer,1);
        json::value op;
        auto const &par = layer.inner_product_param();
        op["name"] = layer.name();
        op["type"] = "InnerProduct";
        bool bias = par.bias_term();
    
        op["params"][0] = param_name(layer,0);
        if(bias)
            op["params"][1] = param_name(layer,1);
        op["inputs"][0] = layer.bottom(0);
        op["outputs"][0] = layer.top(0);

        json::value &opt = op["options"];
        opt["bias"] = bias;
        opt["outputs"] = par.num_output();
        DLPRIM_CHECK(par.axis() == 1 || par.axis() == -1);
        DLPRIM_CHECK(par.transpose() == false);

        add_operator(layer,op,2);

    }
    void CaffeModel::add_input(caffe::LayerParameter const &layer)
    {
        auto const &ip = layer.input_param();
        DLPRIM_CHECK(layer.top_size() >= 1);
        DLPRIM_CHECK(layer.top_size() == ip.shape_size());
        for(int i=0;i<layer.top_size();i++) {
            std::vector<int> dims(ip.shape(i).dim().begin(),ip.shape(i).dim().end());
            json::value inp;
            inp["name"] = layer.top(i);
            inp["shape"] = dims;
            d->net["inputs"].array().push_back(inp);
            d->edges[layer.top(i)] = dims.size();
        }
    }
    void CaffeModel::add_flatten(caffe::LayerParameter const &layer)
    {
        check_inputs(layer,1);
        check_outputs(layer,1);
        DLPRIM_CHECK(layer.flatten_param().axis() == 1 && layer.flatten_param().end_axis()==-1);
        json::value op;
        op["name"] = layer.name();
        op["type"] = "Flatten";
        op["inputs"][0] = layer.bottom(0);
        op["outputs"][0] = layer.top(0);
        add_operator(layer,op,2);
    }
    void CaffeModel::add_reshape(caffe::LayerParameter const &layer)
    {
        check_inputs(layer,1);
        check_outputs(layer,1);
        auto const &rs = layer.reshape_param().shape();
        std::vector<int> shape(rs.dim().begin(),rs.dim().end());
        DLPRIM_CHECK(layer.reshape_param().axis() == 0 && layer.reshape_param().num_axes()==-1);
        json::value op;
        op["name"] = layer.name();
        op["type"] = "Reshape";
        op["inputs"][0] = layer.bottom(0);
        op["outputs"][0] = layer.top(0);
        op["options"]["dims"] = shape;
        add_operator(layer,op,shape.size());
    }
    void CaffeModel::add_concat(caffe::LayerParameter const &layer)
    {
        DLPRIM_CHECK(layer.bottom_size() > 0);
        check_inputs(layer,layer.bottom_size());
        check_outputs(layer,1);
        json::value op;
        op["name"] = layer.name();
        op["type"] = "Concat";
        op["inputs"] = std::vector<std::string>(layer.bottom().begin(),layer.bottom().end());
        op["outputs"][0] = layer.top(0);
        op["options"]["dim"] = layer.concat_param().axis();
        add_operator(layer,op,d->edges[layer.bottom(0)]);

    }
    void CaffeModel::add_slice(caffe::LayerParameter const &layer)
    {
        check_inputs(layer,1);
        int N = layer.slice_param().slice_point_size() + 1;
        check_outputs(layer,layer.slice_param().slice_point_size() + 1);
        for(int i=0;i<N;i++) {
            int start=0,end = -1;
            if(i > 0)
                start = layer.slice_param().slice_point(i-1);
            if(i < N - 1)
                end = layer.slice_param().slice_point(i);
            json::value op;
            op["name"] = layer.name() + "_DLP_" + std::to_string(i);
            op["type"] = "Slice";
            op["inputs"][0] = layer.bottom(0);
            op["outputs"][0] = layer.top(i);
            op["options"]["dim"] = layer.slice_param().axis();
            op["options"]["begin"] = start;
            op["options"]["end"] = end;
            add_operator(layer,op,false);
            d->edges[layer.top(i)] = d->edges[layer.bottom(0)];
        }
    }
    std::string CaffeModel::non_inplace_input(caffe::LayerParameter const &layer)
    {
        if(is_inplace(layer))
            return remove_inplace(layer.bottom(0));
        return layer.bottom(0);
    }
    void CaffeModel::add_threshold(caffe::LayerParameter const &layer)
    {
        check_inputs(layer,1);
        check_outputs(layer,1,1,true);
        json::value op;
        op["name"] = layer.name();
        op["type"] = "Threshold";
        std::string input_name = non_inplace_input(layer);
        op["inputs"][0] = input_name;
        op["outputs"][0] = layer.top(0);
        op["options"]["threshold"] = layer.threshold_param().threshold();
        add_operator(layer,op,d->edges[input_name]);
    }
    void CaffeModel::add_abs(caffe::LayerParameter const &layer)
    {
        check_inputs(layer,1);
        check_outputs(layer,1,1,true);
        json::value op;
        op["name"] = layer.name();
        op["type"] = "Abs";
        op["inputs"][0] = non_inplace_input(layer);
        op["outputs"][0] = layer.top(0);
        add_operator(layer,op,d->edges[layer.bottom(0)]);
    }
    void CaffeModel::add_reduction(caffe::LayerParameter const &layer)
    {
        check_inputs(layer,1);
        check_outputs(layer,1);
        json::value op;
        op["name"] = layer.name();
        op["type"] = "Reduction";
        op["inputs"][0] = layer.bottom(0);
        op["outputs"][0] = layer.top(0);
        auto const &pr = layer.reduction_param();
        op["options"]["output_scale"] = pr.coeff();
        op["options"]["start_axis"] = pr.axis();
        op["options"]["keep_dim"] = false;
        std::string method;
        switch(pr.operation()) {
        case caffe::ReductionParameter::SUM:   method = "sum";     break;
        case caffe::ReductionParameter::ASUM:  method = "abssum";  break;
        case caffe::ReductionParameter::SUMSQ: method = "sumsq";   break;
        case caffe::ReductionParameter::MEAN:  method = "mean";    break;
        default:
            throw ValidationError("Unsupported reduction");
        }
        op["operation"]["method"] = method;
        int input_dim = d->edges[layer.bottom(0)];
        int axis = pr.axis();
        if(axis < 0)
            axis = input_dim + axis;
        int N = std::max(1,axis);
        add_operator(layer,op,N);
    }
    void CaffeModel::add_layers()
    {
        for(int i=0;i<d->model.layer_size();i++) {
            caffe::LayerParameter const &layer = d->model.layer(i);
            std::string type = layer.type();
            
            #if 0
            {
                std::string s;
                google::protobuf::TextFormat::PrintToString(layer,&s);
                std::cout << "Loaded:\n" << s << std::endl;
            }
            #endif

            if(type == "Convolution" || type == "Deconvolution")
                add_conv(layer);
            else if(type == "BatchNorm")
                add_bn(layer);
            else if(type == "Scale")
                add_scale(layer);
            else if(type == "Eltwise")
                add_elementwise(layer);
            else if(type == "ReLU")
                add_standard_activation(layer,"relu");
            else if(type == "TanH")
                add_standard_activation(layer,"tanh");
            else if(type == "Sigmoid")
                add_standard_activation(layer,"sigmoid");
            else if(type == "AbsVal")
                add_abs(layer);
            else if(type == "Threshold")
                add_threshold(layer);
            else if(type == "Pooling")
                add_pool2d(layer);
            else if(type == "Softmax")
                add_softmax(layer);
            else if(type == "InnerProduct")
                add_ip(layer);
            else if(type == "Slice")
                add_slice(layer);
            else if(type == "Concat")
                add_concat(layer);
            else if(type == "Flatten")
                add_flatten(layer);
            else if(type == "Reshape")
                add_reshape(layer);
            else if(type == "Reduction")
                add_reduction(layer);
            else if(type == "Input")
                add_input(layer);
            else if(type == "Silence") 
                ; // meta layer nothing to do
            else if(type == "Dropout")
                add_standard_activation(layer,"identity");
            else
                throw ValidationError("Unsupported layer type " + type); 

        }
    }
    json::value const &CaffeModel::network()
    {
        return d->net;
    }
    ///
    /// Query parameter by name, if not found empty/null tensor returned
    ///
    Tensor CaffeModel::get_parameter(std::string const &name)
    {
        auto p=d->parameters.find(name);
        if(p == d->parameters.end())
            return Tensor();
        return p->second;
    }

    void CaffeModel::load_text_proto(std::string const &file_name)
    {
        std::ostringstream ss;
        std::ifstream f(file_name);
        ss << f.rdbuf();
        if(!google::protobuf::TextFormat::ParseFromString(ss.str(),&d->model))
            throw ValidationError("Failed to parse " + file_name);
    }

    void CaffeModel::load_binary_proto(std::string const &file_name)
    {
        std::ifstream f(file_name,std::ifstream::binary);
        f.seekg(0,std::ifstream::end);
        size_t length = f.tellg();
        if(length >= (1u<<31) || length == 0)
            throw ValidationError("File " + file_name + " has invalid size");
        f.seekg(0);
        std::vector<char> buffer(length);
        f.read(buffer.data(),length);
        if(!f)
            throw ValidationError("IO Error reading " + file_name);
        google::protobuf::io::CodedInputStream stream(reinterpret_cast<unsigned char *>(buffer.data()),length);
        stream.SetTotalBytesLimit(std::numeric_limits<int>::max(),std::numeric_limits<int>::max());
        if(!d->params.MergePartialFromCodedStream(&stream))
            throw ValidationError("Protobuf Parsing Error " + file_name);
    }
}
