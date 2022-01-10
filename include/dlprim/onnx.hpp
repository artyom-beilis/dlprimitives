#pragma once
#include <dlprim/model.hpp>
namespace onnx {
    class TensorProto;
    class NodeProto;
}
namespace dlprim {
    
    ///
    /// External model for loading ONNX models for inference with dlprim
    ///
    /// Use example:
    /// 
    ///     dlprim::Context ctx(device_name);
    ///     model.load(path_to_onnx_file);
    ///     dlprim::Net net(ctx);
    ///     net.load_model(model);
    ///
    class ONNXModel : public ModelBase {
    public:
        ONNXModel();
        virtual ~ONNXModel();
        ///
        /// Parse and prepare the model from ONNX file
        ///
        void load(std::string const &file_name);
        ///
        /// Generated network
        ///
        virtual json::value const &network();
        ///
        /// Query parameter by name, if not found empty/null tensor returned
        ///
        virtual Tensor get_parameter(std::string const &name);
    private:
        void load_proto(std::string const &file_name);
        void prepare_network();
        void prepare_inputs_outputs();
        void parse_operators();
        void validate_outputs();
        void add_conv(onnx::NodeProto const &node);
        void add_ip(onnx::NodeProto const &node);
        void add_bn(onnx::NodeProto const &node);
        void add_operator(onnx::NodeProto const &node,json::value &v,bool add_outputs = true);
        void add_standard_activation(onnx::NodeProto const &node,std::string const &name);
        void add_concat(onnx::NodeProto const &node);
        void add_softmax(onnx::NodeProto const &node);
        void add_elementwise(onnx::NodeProto const &node,std::string const &operation);
        void add_global_pooling(onnx::NodeProto const &node,std::string const &operation);
        void add_pool2d(onnx::NodeProto const &node,std::string const &operation);
        void add_flatten(onnx::NodeProto const &node);

        void check_outputs(onnx::NodeProto const &node,int minv,int maxv=-1);
        void check_inputs(onnx::NodeProto const &node,int inputs_min,int inputs_max=-1,int params_min=0,int params_max=-1);

        std::vector<int> get_pads(onnx::NodeProto const &node);
        template<typename T>
        T get_attr(onnx::NodeProto const &node,std::string const &name,T default_value);
        template<typename T>
        T get_attr(onnx::NodeProto const &node,std::string const &name);
        std::pair<Tensor,std::string> to_tensor(onnx::TensorProto const &init);
        struct Data;
        std::unique_ptr<Data> d;
    };

}
