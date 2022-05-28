///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
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
    class DLPRIM_API ONNXModel : public ModelBase {
    public:
        ONNXModel();
        virtual ~ONNXModel();
        ///
        /// Parse and prepare the model from ONNX file
        ///
        void load(std::string const &file_name);
        
        ///
        /// get input shapes so dynamic shapes can be fixed to their limits, the 
        /// dynamix axis has dimension 0.
        ///
        std::map<std::string,Shape> input_shapes() const;

        /// return list of dynamix axes index per input tensor
        std::map<std::string,std::vector<int> > dynamic_axes() const;

        ///
        /// Set initial dimension for dynamic axis
        ///
        void set_dynamic_axis(std::string const &name,int axis,size_t limit);

        ///
        /// Modify first (batch) dynamix axis of each input shape
        ///
        void set_batch(size_t size);
        

        ///
        /// Generates the network, called after the dynamic axis are defined
        ///
        void build();

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
        void add_matmul(onnx::NodeProto const &node);
        void add_bn(onnx::NodeProto const &node);
        void add_operator(onnx::NodeProto const &node,json::value &v,bool add_outputs = true);
        void add_standard_activation(onnx::NodeProto const &node,std::string const &name,bool validate_inputs = true);
        void add_concat(onnx::NodeProto const &node);
        void add_softmax(onnx::NodeProto const &node);
        void add_elementwise(onnx::NodeProto const &node,std::string const &operation);
        void add_global_pooling(onnx::NodeProto const &node,std::string const &operation);
        void add_pool2d(onnx::NodeProto const &node,std::string const &operation);
        void add_flatten(onnx::NodeProto const &node);
        void add_clip(onnx::NodeProto const &node);
        void add_pad(onnx::NodeProto const &node);
        void add_bias(onnx::NodeProto const &node);
        void add_squeeze(onnx::NodeProto const &node);
        void add_reshape(onnx::NodeProto const &node);
        void add_reduce(onnx::NodeProto const &node,std::string const &method);
        void handle_constant(onnx::NodeProto const &node);
        std::pair<std::string,Tensor> transpose_parameter(std::string const &name);


        template<typename T>
        T get_scalar_constant(std::string const &name);

        void check_outputs(onnx::NodeProto const &node,int minv,int maxv=-1);
        void check_inputs(onnx::NodeProto const &node,int inputs_min,int inputs_max=-1,int params_min=0,int params_max=-1);

        std::vector<int> tensor_to_intvec(Tensor t);
        void check_pad_op_2d(onnx::NodeProto const &node,std::vector<int> &pads);
        std::vector<int> get_pads_2d(onnx::NodeProto const &node);
        bool has_attr(onnx::NodeProto const &node,std::string const &name);
        template<typename T>
        T get_attr(onnx::NodeProto const &node,std::string const &name,T default_value);
        template<typename T>
        T get_attr(onnx::NodeProto const &node,std::string const &name);
        struct Data;
        std::unique_ptr<Data> d;
    };

}
