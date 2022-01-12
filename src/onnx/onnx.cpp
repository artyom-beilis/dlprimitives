#include <dlprim/onnx.hpp>
#include <dlprim/json.hpp>

#include "onnx.pb.h"

#include <fstream>
#include <set>

#include <google/protobuf/io/coded_stream.h>

namespace dlprim {
    struct ONNXModel::Data {
        onnx::ModelProto model;
        json::value net;
        std::map<std::string,Tensor> parameters;
        std::set<std::string> edges;
        std::map<std::string,Tensor> constants;
    };
    namespace {
        std::pair<Tensor,std::string> to_tensor(onnx::TensorProto const &init)
        {
            std::string const &name = init.name();
            Shape sp=Shape::from_range(init.dims().begin(),init.dims().end());
            DataType dt = float_data;
            switch(init.data_type()) {
            case onnx::TensorProto::FLOAT:
                dt = float_data;
                break;
            case onnx::TensorProto::INT64:
                dt = int64_data;
                break;
            default:
                {
                    std::ostringstream ss;
                    ss << "Unsupported data type " << init.data_type() << " for tensor with shape " << sp << " name " << name;
                    throw ValidationError(ss.str());
                }
            }
            if(sp.size() == 0)
                sp = Shape(1);
            Context ctx; // cpu context
            Tensor result(ctx,sp,dt);
            size_t size = sp.total_size();
            size_t memsize = result.memory_size();
            void *vptr = result.host_data();
            if(init.has_raw_data() && init.raw_data().size() == memsize) {
                memcpy(vptr,init.raw_data().c_str(),memsize);
            }
            else if(dt == float_data && size_t(init.float_data().size()) == size) {
                float *ptr = result.data<float>();
                for(float const &v : init.float_data())
                    *ptr++ = v;
            }
            else if(dt == int64_data && size_t(init.int64_data().size()) == size) {
                std::int64_t *ptr = result.data<std::int64_t>();
                for(int64_t const &v : init.int64_data()) {
                    *ptr++ = v;
                }
            }
            else {
                throw ValidationError("No info found for tensor");
            }
            return std::make_pair(result,name);
        }
    }

    ONNXModel::ONNXModel() : d(new Data())
    {
    }
    ONNXModel::~ONNXModel() {}

    json::value const &ONNXModel::network()
    {
        return d->net;
    }

    Tensor ONNXModel::get_parameter(std::string const &name)
    {
        auto p=d->parameters.find(name);
        if(p == d->parameters.end())
            return Tensor();
        return p->second;
    }

    void ONNXModel::load_proto(std::string const &file_name)
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
        if(!d->model.MergePartialFromCodedStream(&stream))
            throw ValidationError("Protobuf Parsing Error " + file_name);
    }

    void ONNXModel::load(std::string const &file_name)
    {
        load_proto(file_name);
        prepare_network();
        validate_outputs();
    }

    void ONNXModel::prepare_network()
    {
        prepare_inputs_outputs();
        parse_operators();
    }
    void ONNXModel::validate_outputs()
    {
        for(auto const &output : d->model.graph().output()) {
            std::string const &name = output.name();
            if(d->edges.find(name) == d->edges.end()) {
                throw ValidationError("No such output " + name + " generated in graph");
            }
        }
    }
    void ONNXModel::prepare_inputs_outputs()
    {
        DLPRIM_CHECK(d->model.has_graph());
        onnx::GraphProto const &graph = d->model.graph();
        json::array inputs,outputs;
       
        for(onnx::TensorProto const &init : graph.initializer()) { 
            auto p=to_tensor(init);
            d->parameters[p.second] = p.first;
        }
        for(onnx::ValueInfoProto const &input : graph.input()) {
            std::string name = input.name();
            if(d->parameters.find(name) != d->parameters.end())
                continue;
            onnx::TypeProto const &type = input.type();
            DLPRIM_CHECK(type.value_case() == onnx::TypeProto::kTensorType);
            onnx::TypeProto_Tensor const &tensor_type = type.tensor_type();
            DLPRIM_CHECK(tensor_type.has_elem_type());
            DLPRIM_CHECK(tensor_type.elem_type() == onnx::TensorProto::FLOAT);
            DLPRIM_CHECK(tensor_type.has_shape());
            json::value jinp;
            jinp["name"] = name;
            jinp["shape"]=json::array();
            int index = 0;
            for(auto const &dv : tensor_type.shape().dim()) {
                jinp["shape"][index++] = dv.dim_value();
            }
            inputs.push_back(jinp);
            d->edges.insert(name);
        }
        for(auto const &output : graph.output()) {
            std::string name = output.name();
            outputs.push_back(name);
        }
        d->net["inputs"]  = inputs;
        d->net["outputs"] = outputs;
    }


    namespace {
        template<typename T>
        struct ParseAttr;

        template<>
        struct ParseAttr<int> {
            static int get(onnx::AttributeProto const &a) 
            {
                DLPRIM_CHECK(a.type() == a.INT);
                return a.i();
            }
        };
        template<>
        struct ParseAttr<double> {
            static double get(onnx::AttributeProto const &a) 
            {
                DLPRIM_CHECK(a.type() == a.FLOAT);
                return a.f();
            }
        };
        template<>
        struct ParseAttr<std::string> {
            static std::string get(onnx::AttributeProto const &a) 
            {
                DLPRIM_CHECK(a.type() == a.STRING);
                return a.s();
            }
        };
        template<>
        struct ParseAttr<std::vector<int> > {
            static std::vector<int> get(onnx::AttributeProto const &a) 
            {
                DLPRIM_CHECK(a.type() == a.INTS);
                std::vector<int> r(a.ints().begin(),a.ints().end());
                return r;
            }
        };
        template<>
        struct ParseAttr<std::vector<double> > {
            static std::vector<int> get(onnx::AttributeProto const &a) 
            {
                DLPRIM_CHECK(a.type() == a.FLOATS);
                std::vector<int> r(a.floats().begin(),a.floats().end());
                return r;
            }
        };
        template<>
        struct ParseAttr<Tensor> {
            static Tensor get(onnx::AttributeProto const &a) 
            {
                DLPRIM_CHECK(a.type() == a.TENSOR);
                return to_tensor(a.t()).first;
            }
        };

    }

    template<typename T>
    T ONNXModel::get_attr(onnx::NodeProto const &node,std::string const &name,T default_value)
    {
        T result = default_value;
        for(auto const &attr : node.attribute()) {
            if(attr.name() == name) {
                result = ParseAttr<T>::get(attr);
                break;
            }
        }
        return result;
    }
    template<typename T>
    T ONNXModel::get_attr(onnx::NodeProto const &node,std::string const &name)
    {
        for(auto const &attr : node.attribute()) {
            if(attr.name() == name) {
                return ParseAttr<T>::get(attr);
                break;
            }
        }
        throw ValidationError("No attribute named " + name + " for op:" + node.name());
    }

    std::vector<int> ONNXModel::get_pads(onnx::NodeProto const &node)
    {
        std::vector<int> pads = get_attr(node,"pads",std::vector<int>{0,0});
        if(pads.size() == 4) {
            DLPRIM_CHECK(pads[0] == pads[2] && pads[1] == pads[3]);
            pads.resize(2);
        }
        return pads;
    }

    void ONNXModel::check_outputs(onnx::NodeProto const &node,int minv,int maxv)
    {
        if(maxv == -1)
            maxv = minv;
        DLPRIM_CHECK(minv <= node.output_size() && node.output_size() <= maxv);
        for(auto const &name:node.output()) {
            DLPRIM_CHECK(d->edges.find(name) == d->edges.end());
        }
    }
    void ONNXModel::check_inputs(onnx::NodeProto const &node,int inputs_min,int inputs_max,int params_min,int params_max)
    {
        if(inputs_max == -1)
            inputs_max = inputs_min;
        if(params_max == -1)
            params_max = params_min;

        int size = node.input_size();
        DLPRIM_CHECK(inputs_min + params_min <= size && size <= inputs_max + params_max);
        int inputs_count = -1;
        if(inputs_min == inputs_max) {
            inputs_count = inputs_min;
        }
        else {
            DLPRIM_CHECK(params_max == params_min);
            inputs_count = size - params_min;
        }
        int index = 0;
        for(;index<inputs_count;index++) {
            DLPRIM_CHECK(d->edges.find(node.input(index))!=d->edges.end());
        }
        for(;index<size;index++) {
            DLPRIM_CHECK(d->parameters.find(node.input(index))!=d->parameters.end());
        }
    }
    
    void ONNXModel::add_conv(onnx::NodeProto const &node)
    {
        check_inputs(node,1,1,1,2);
        check_outputs(node,1);
        auto pW = d->parameters.find(node.input(1));
        Tensor &W = pW->second;
        if(W.shape().size() != 4)
            throw ValidationError("Only 2d conv is supported");
        bool bias = node.input_size() == 3;
        json::value v;
        v["name"] = node.name();
        v["type"] = "Convolution2D",
        v["inputs"][0] = node.input(0);
        v["outputs"][0] = node.output(0);
        v["params"][0] = node.input(1);
        if(bias)
            v["params"][1] = node.input(2);

        json::value &opt=v["options"];
        DLPRIM_CHECK(get_attr<std::string>(node,"auto_pad","NOTSET")=="NOTSET");
        int groups = get_attr(node,"group",1);
        opt["bias"] = bias;
        opt["groups"] = groups;
        opt["kernel"] = get_attr(node,"kernel_shape",std::vector<int>{int(W.shape()[2]),int(W.shape()[3])});
        opt["stride"] = get_attr(node,"strides",std::vector<int>{1,1});
        opt["dilate"] = get_attr(node,"dilations",std::vector<int>{1,1});
        opt["pad"] = get_pads(node);
        opt["channels_out"]=W.shape()[0];
        opt["channels_in"]=W.shape()[1] * groups;

        add_operator(node,v);
    }
    void ONNXModel::add_ip(onnx::NodeProto const &node)
    {
        DLPRIM_CHECK(get_attr<double>(node,"alpha") == 1.0);
        DLPRIM_CHECK(get_attr<double>(node,"beta") == 1.0);
        DLPRIM_CHECK(get_attr<int>(node,"transA",0) == 0);
        DLPRIM_CHECK(get_attr<int>(node,"transB",0) == 1);
        check_inputs(node,1,1,1,2);
        check_outputs(node,1);
        bool bias = node.input_size() == 3;
        Tensor &W = d->parameters[node.input(1)];
        json::value op;
        op["name"] = node.name();
        op["type"] = "InnerProduct";
        op["inputs"][0] = node.input(0);
        op["outputs"][0] = node.output(0);
        op["params"][0] = node.input(1);
        if(bias)
            op["params"][1] = node.input(2);
        json::value &opt = op["options"];
        opt["bias"] = bias;
        opt["inputs"] = W.shape()[1];
        opt["outputs"] = W.shape()[0];
        add_operator(node,op);
    }
    
    void ONNXModel::add_bn(onnx::NodeProto const &node)
    {
        check_inputs(node,1,1,4,4);
        check_outputs(node,1,5);
        double eps = get_attr<double>(node,"epsilon");
        double momentum = 1.0 - get_attr<double>(node,"momentum");
        json::value op;
        op["name"] = node.name();
        op["type"] = "BatchNorm";
        op["inputs"][0] = node.input(0);
        op["outputs"][0] = node.output(0);
        op["params"] = std::vector<std::string>{ node.input(3), node.input(4), node.input(1), node.input(2) };
        op["options"]["eps"] = eps;
        op["options"]["momentum"] = momentum;

        add_operator(node,op,false);
        /// don't add special mean ops
        d->edges.insert(node.output(0));
    }
    void ONNXModel::add_operator(onnx::NodeProto const &node,json::value &v,bool add_outputs)
    {
        d->net["operators"].array().push_back(std::move(v));
        if(add_outputs) {
            for(std::string const &output : node.output()) {
                d->edges.insert(output);
            }
        }
    }


    void ONNXModel::add_standard_activation(onnx::NodeProto const &node,std::string const &name,bool validate_inputs)
    {
        if(validate_inputs) {
            check_inputs(node,1);
            check_outputs(node,1);
        }
        bool is_mergable = false;
        std::string type;
        json::array &ops = d->net["operators"].array();
        if( !ops.empty() 
            && ((type=ops.back()["type"].str()) == "Convolution2D" || type == "InnerProduct" || type == "Elementwise")
            && ops.back()["options"].get("activation","identity") == "identity"
            && ops.back()["outputs"][0] == node.input(0))
        {
            is_mergable = true;
            for(onnx::NodeProto const &n : d->model.graph().node()) {
                if(!is_mergable)
                    break;
                if(&n == &node)
                    continue;
                for(auto const &in : n.input()) {
                    if(in == node.input(0)) {
                        is_mergable = false;
                        break;
                    }
                }
            }
            for(auto const &output: d->model.graph().output()) {
                if(output.name() == node.input(0)) {
                    is_mergable = false;
                    break;
                }
            }
        }
        if(is_mergable) {
            d->edges.erase(node.input(0));
            d->edges.insert(node.output(0));
            ops.back()["outputs"][0] = node.output(0);
            ops.back()["options"]["activation"] = name;
        }
        else {
            json::value op;
            op["name" ] = node.name();
            op["type" ] = "Activation";
            op["inputs"][0] = node.input(0);
            op["outputs"][0] = node.output(0);
            op["options"]["activation"] = name;
            add_operator(node,op);
        }
    }
    void ONNXModel::add_concat(onnx::NodeProto const &node)
    {
        check_inputs(node,1,std::numeric_limits<int>::max());
        check_outputs(node,1);
        json::value op;
        op["name"] = node.name();
        op["type"] = "Concat";
        op["inputs"] = std::vector<std::string>(node.input().begin(),node.input().end());
        op["outputs"][0] = node.output(0);
        op["options"]["dim"] = get_attr<int>(node,"axis");
        add_operator(node,op);
    }
    void ONNXModel::add_softmax(onnx::NodeProto const &node)
    {
        check_inputs(node,1);
        check_outputs(node,1);
        int attr = get_attr(node,"axis",-1);
        DLPRIM_CHECK(attr == 1 || attr == -1); 
        json::value op;
        op["name"] = node.name();
        op["type"] = "Concat";
        op["inputs"][0] = node.input(0);
        op["outputs"][0] = node.output(0);
        add_operator(node,op);
    }
    void ONNXModel::add_elementwise(onnx::NodeProto const &node,std::string const &operation)
    {
        check_inputs(node,2);
        check_outputs(node,1);
        json::value op;
        op["name"] = node.name();
        op["type"] = "Elementwise";
        op["inputs"] = std::vector<std::string>(node.input().begin(),node.input().end());
        op["outputs"][0] = node.output(0);
        op["options"]["operation"] = operation;
        add_operator(node,op);
    }
    
    void ONNXModel::add_global_pooling(onnx::NodeProto const &node,std::string const &operation)
    {
        check_inputs(node,1);
        check_outputs(node,1);
        json::value op;
        op["name"] = node.name();
        op["type"] = "GlobalPooling";
        op["inputs"][0] = node.input(0);
        op["outputs"][0] = node.output(0);
        op["options"]["mode"] = operation;
        add_operator(node,op);
    }

    void ONNXModel::add_pool2d(onnx::NodeProto const &node,std::string const &operation)
    {
        check_inputs(node,1);
        check_outputs(node,1);
        json::value op;
        DLPRIM_CHECK((get_attr(node,"dilations",std::vector<int>{1,1})==std::vector<int>{1,1}));
        DLPRIM_CHECK(get_attr<std::string>(node,"auto_pad","NOTSET") == "NOTSET"); 
        auto kern = get_attr<std::vector<int> >(node,"kernel_shape");
        bool ceil_mode = get_attr(node,"ceil_mode",0);
        auto strides = get_attr<std::vector<int> >(node,"strides");
        auto pads = get_attr(node,"pads",std::vector<int>{0,0});
        // for opset 9 simulation of external padding
        if( pads.size() == 4 
            && pads[0] == 0 && pads[1] == 0 
            && pads[2] >  0 && pads[2] > 0
            && strides[0] == 2 && strides[1] == 2
            && !ceil_mode)
        {
            ceil_mode = true;
            pads.resize(2);
        }
        else {
            pads = get_pads(node);
        }

        bool count_include_pad = get_attr(node,"count_include_pad",0);
        op["name"] = node.name();
        op["type"] = "Pooling2D";
        op["inputs"][0] = node.input(0);
        op["outputs"][0] = node.output(0);
        json::value &opt = op["options"];
        opt["mode"] = operation;
        opt["kernel"] = kern;
        opt["stride"] = strides;
        opt["pad"] = pads;
        opt["count_include_pad"] = count_include_pad;
        opt["ceil_mode"] = ceil_mode;
        add_operator(node,op);
    }

    void ONNXModel::add_flatten(onnx::NodeProto const &node)
    {
        check_inputs(node,1);
        check_outputs(node,1);
        DLPRIM_CHECK(get_attr<int>(node,"axis")==1);
        json::value op;
        op["name"] = node.name();
        op["type"] = "Flatten";
        op["inputs"][0] = node.input(0);
        op["outputs"][0] = node.output(0);
        add_operator(node,op);
    }
    
    void ONNXModel::handle_constant(onnx::NodeProto const &node)
    {
        check_inputs(node,0);
        check_outputs(node,1);
        Tensor t=get_attr<Tensor>(node,"value");
        d->constants[node.output(0)] = t;
    }

    template<typename T>
    T ONNXModel::get_scalar_constant(std::string const &name)
    {
        auto p = d->constants.find(name);
        if(p == d->constants.end())
            throw ValidationError("No such constant " + name);
        if(p->second.shape() != Shape(1))
            throw ValidationError("Invalid constant Shape,expecting (1)");
        return p->second.data<T>()[0];
    }

    void ONNXModel::add_clip(onnx::NodeProto const &node)
    {
        float min_val = get_attr<double>(node,"min",std::numeric_limits<float>::min());
        float max_val = get_attr<double>(node,"max",std::numeric_limits<float>::min());
        DLPRIM_CHECK(1<= node.input_size() && node.input_size() <= 3);
        DLPRIM_CHECK(d->edges.count(node.input(0)) == 1);
        if(node.input_size() >= 2) {
            min_val = get_scalar_constant<float>(node.input(1));
        }
        if(node.input_size() == 3) {
            max_val = get_scalar_constant<float>(node.input(2));
        }
        
        if(max_val == 6 && min_val == 0)  {
            add_standard_activation(node,"relu6",false);
            return;
        }

        json::value op;
        
        op["name"] = node.name();
        op["type"] = "Hardtanh";
        op["inputs"][0] = node.input(0);
        op["outputs"][0] = node.output(0);
        json::value &opt = op["options"];
        opt["min_val"] = min_val;
        opt["max_val"] = max_val;

        add_operator(node,op);
    }

    void ONNXModel::add_pad(onnx::NodeProto const &node)
    {
        std::vector<int> pads;
        check_outputs(node,1);
        DLPRIM_CHECK(node.input_size() >= 1);
        DLPRIM_CHECK(d->edges.count(node.input(0)) == 1);
        if(node.input_size() >= 2) {
            auto p = d->constants.find(node.input(1));
            DLPRIM_CHECK(p!= d->constants.end());
            size_t n=p->second.shape().total_size();
            if(p->second.dtype() == int64_data) {
                int64_t *values = p->second.data<int64_t>();
                for(size_t i=0;i<n;i++)
                    pads.push_back(values[i]);
            }
            else if(p->second.dtype() == int32_data) {
                int32_t *values = p->second.data<int32_t>();
                for(size_t i=0;i<n;i++)
                    pads.push_back(values[i]);
            }
            else {
                throw ValidationError("Pads input should have only int32/int64 type");
           }
            
        }
        else {
            pads = get_attr<std::vector<int> >(node,"pads",std::vector<int>());
        }
        for(auto v: pads) {
            if(v != 0) {
                throw ValidationError("Pad operator supported for now only as dummy operator with 0 pads only");
            }
        }
        add_standard_activation(node,"identity",false);
    }

    void ONNXModel::parse_operators()
    {
        d->net["operators"] = json::array();
        for(onnx::NodeProto const &node : d->model.graph().node()) {
            DLPRIM_CHECK(node.has_op_type());
            std::string op = node.op_type();
            
            if(op == "Conv")
                add_conv(node);
            else if(op == "Gemm")
                add_ip(node);
            else if(op == "BatchNormalization")
                add_bn(node);
            else if(op == "Clip") 
                add_clip(node);
            else if(op == "Relu")
                add_standard_activation(node,"relu");
            else if(op == "Sigmoid")
                add_standard_activation(node,"sigmoid"); 
            else if(op == "Tanh")
                add_standard_activation(node,"tanh"); 
            else if(op == "Concat")
                add_concat(node);
            else if(op == "Softmax")
                add_softmax(node);
            else if(op == "Mul")
                add_elementwise(node,"prod");
            else if(op == "Add")
                add_elementwise(node,"sum");
            else if(op == "GlobalAveragePool")
                add_global_pooling(node,"avg");
            else if(op == "GlobalMaxPool")
                add_global_pooling(node,"max");
            else if(op == "MaxPool")
                add_pool2d(node,"max");
            else if(op == "AveragePool")
                add_pool2d(node,"avg");
            else if(op == "Flatten") 
                add_flatten(node);
            else if(op == "Pad")
                add_pad(node);
            else if(op == "Constant")
                handle_constant(node);
            else
                throw ValidationError("Unsupported ONNX Operator:" + op);
        }
    }

}
