///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/model.hpp>
namespace caffe_dlprim {
    class LayerParameter;
    class BlobProto;
    class ConvolutionParameter;
}

namespace caffe = caffe_dlprim;
namespace dlprim {
    
    ///
    /// External model for loading Caffe models for inference with dlprim
    ///
    /// Use example:
    /// 
    ///     dlprim::Context ctx(device_name);
    ///     model.load(path_to_caffe_proto,path_to_caffe_params);
    ///     dlprim::Net net(ctx);
    ///     net.load_model(model);
    ///
    class DLPRIM_API CaffeModel : public ModelBase {
    public:
        CaffeModel();
        virtual ~CaffeModel();
        ///
        /// Parse and prepare the model from Caffe file
        ///
        void load(std::string const &proto_file_name,std::string const &caffe_model);
        ///
        /// Generated network
        ///
        virtual json::value const &network();
        ///
        /// Query parameter by name, if not found empty/null tensor returned
        ///
        virtual Tensor get_parameter(std::string const &name);
    private:
        Tensor to_tensor(caffe::BlobProto const &blob);
        void get_outputs();
        void check_inputs(caffe::LayerParameter const &layer,int inputs_min,int inputs_max=-1);
        void check_outputs(caffe::LayerParameter const &layer,int minv,int maxv=-1,bool allow_inplace=false);
        std::string param_name(caffe::LayerParameter const &layer,int index,bool check=true);
        std::string param_name(std::string const &n,int index);
        std::string non_inplace_input(caffe::LayerParameter const &layer);
        void load_binary_proto(std::string const &file_name);
        void load_text_proto(std::string const &file);
        void load_parameters();
        void add_operator(caffe::LayerParameter const &layer,json::value &v,int dims,bool add_outputs=true);
        void add_deprecated_input();
        void add_layers();
        void get_conv_params(caffe::ConvolutionParameter const &lp,json::value &opt);
        void add_standard_activation(caffe::LayerParameter const &layer,std::string const &name,bool validate_inputs=true);

        void add_conv(caffe::LayerParameter const &layer);
        void add_ip(caffe::LayerParameter const &layer);
        void add_bn(caffe::LayerParameter const &layer);
        void add_abs(caffe::LayerParameter const &layer);
        void add_threshold(caffe::LayerParameter const &layer);
        void add_scale(caffe::LayerParameter const &layer);
        void add_pool2d(caffe::LayerParameter const &layer);
        void add_softmax(caffe::LayerParameter const &layer);
        void add_input(caffe::LayerParameter const &layer);
        void add_slice(caffe::LayerParameter const &layer);
        void add_flatten(caffe::LayerParameter const &layer);
        void add_reshape(caffe::LayerParameter const &layer);
        void add_concat(caffe::LayerParameter const &layer);
        void add_reduction(caffe::LayerParameter const &layer);


        std::string remove_inplace(std::string const &name);
        void add_elementwise(caffe::LayerParameter const &layer);
        bool is_mergable(caffe::LayerParameter const &layer);
        json::value *get_input_generator(std::string const &name);
        bool is_inplace(caffe::LayerParameter const &layer);
        struct Data;
        std::unique_ptr<Data> d;
    };

}
