#pragma once
#include <dlprim/model.hpp>
namespace onnx {
    class TensorProto;
}
namespace dlprim {
    
    class ONNXModel : public ModelBase {
    public:
        ONNXModel();
        virtual ~ONNXModel();
        void load(std::string const &file_name);
        virtual json::value const &network();
        virtual Tensor get_parameter(std::string const &name);
    private:
        void load_proto(std::string const &file_name);
        void prepare_network();
        std::pair<Tensor,std::string> to_tensor(onnx::TensorProto const &init);
        struct Data;
        std::unique_ptr<Data> d;
    };

}
