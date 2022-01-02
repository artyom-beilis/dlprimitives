#include <dlprim/onnx.hpp>
#include <dlprim/json.hpp>

#include "onnx.pb.h"

#include <fstream>

namespace dlprim {
    struct ONNXModel::Data {
        onnx::ModelProto model;
        json::value net;
        std::map<std::string,Tensor> parameters;
    };

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
        return Tensor();
    }

    void ONNXModel::load_proto(std::string const &file_name)
    {
        std::ifstream f(file_name,std::ifstream::binary);
        f.seekg(std::ifstream::end);
        size_t length = f.tellg();
        if(length >= (1u<<31) || length == 0)
            throw ValidationError("File " + file_name + " has invalid size");
        f.seekg(0);
        std::vector<char> buffer(length);
        f.read(buffer.data(),length);
        if(!f)
            throw ValidationError("IO Error reading " + file_name);
        d->model.ParseFromArray(buffer.data(),length);
    }

    void ONNXModel::load(std::string const &file_name)
    {
        load_proto(file_name);
        prepare_network();
    }

    std::pair<Tensor,std::string> ONNXModel::to_tensor(onnx::TensorProto const &init)
    {
        std::string const &name = init.name();
        Shape sp=Shape::from_range(init.dims().begin(),init.dims().end());
        DataType dt = float_data;
        switch(init.data_type()) {
        case onnx::TensorProto::FLOAT:
            dt = float_data;
            break;
        default:
            throw ValidationError("Only float is supported so far");
        }
        Context ctx; // cpu context
        Tensor result(ctx,sp,dt);
        size_t size = sp.total_size();
        float *ptr = result.data<float>();
        if(size_t(init.float_data().size()) == size) {
            for(float const &v : init.float_data())
                *ptr++ = v;
        }
        else if(init.has_raw_data() && init.raw_data().size() == result.memory_size()) {
            memcpy(ptr,init.raw_data().c_str(),result.memory_size());
        }
        return std::make_pair(result,name);
    }
    void ONNXModel::prepare_network()
    {
        DLPRIM_CHECK(d->model.has_graph());
        onnx::GraphProto const &graph = d->model.graph();
        json::array inputs,outputs;
       
        for(onnx::TensorProto const &init : graph.initializer()) { 
            auto p=to_tensor(init);
            d->parameters[p.second] = p.first;
        }
    }

}
