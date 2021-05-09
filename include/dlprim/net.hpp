#pragma once
#include <dlprim/operators.hpp>

namespace dlprim {
    class Net {
    public:
        Net(Context &ctx);
        void add_input_tensor(std::string const &name,Shape const &s,DataType d=float_data);
        void add_sequential_operator(std::unique_ptr<Operator> op,std::string name=std::string());
        void add_operator(std::unique_ptr<Operator> op,std::vector<std::string> inputs,std::vector<std::string> outputs);
        void build(int flags = forward_data | backward_data | backward_param);
        std::vector<std::unique_ptr<Operator> > &operators();
        std::vector<std::string> &data_tensor_names();
        std::vector<Tensor> &data_tensors();
        std::vector<Tensor> &diff_tensors();
        void forward();
        void backward();
    private:
    };
};
