#pragma once
#include <dlprim/operator.hpp>
#include <map>
#include <vector>

namespace dlprim {
    class Net {
    public:

        Net(Context &ctx);
        ~Net() {}
        Net const &operator=(Net const &) = delete;
        Net(Net const &) = delete;
        
        Net &operator=(Net &&) = default;
        Net(Net &&) = default;

        void load_from_json(json::value const &v);
        void load_from_json_file(std::string const &name);

        void add_input_tensor(std::string const &name,TensorSpecs const &ts);
        void mark_output_tensor(std::string const &name);

        void load_parameters_from_hdf5(std::string const &fname,bool allow_missing=false);

        void add_operator(  std::unique_ptr<Operator> op,
                            std::string const &name,
                            std::vector<std::string> const &inputs,
                            std::vector<std::string> const &outputs,
                            std::vector<std::string> const &parameters = std::vector<std::string>());

        void mode(CalculationsMode mode); 
        void setup();
        void reshape();
        void copy_parameters_to_device(); 
        void clear_memory();

        std::map<std::string,Tensor> &tensors()
        {
            return tensors_;
        }
        std::map<std::string,Tensor> &tensor_diffs()
        {
            return tensors_diff_;
        }

        std::map<std::string,Tensor> &params()
        {
            return parameters_;
        }
        std::map<std::string,Tensor> &param_diffs()
        {
            return parameters_diff_;
        }

        Tensor &tensor(std::string const &name)
        {
            auto p=tensors_.find(name);
            if(p == tensors_.end())
                throw ValidationError("Unknown tensor name:" + name);
            return p->second;
        }
        
        Tensor &param(std::string const &name)
        {
            auto p=parameters_.find(name);
            if(p == parameters_.end())
                throw ValidationError("Unknown parameter name:" + name);
            return p->second;
        }

        void forward(ExecutionContext const &ectx);
        void backward(ExecutionContext const &ectx);

        std::vector<std::string> const &input_names()
        {
            return inputs_;
        }

        std::vector<std::string> const &output_names()
        {
            return outputs_;
        }

        Tensor &input(unsigned id)
        {
            if(id >= inputs_.size())
                throw ValidationError("Invalid input id");
            return tensor(inputs_[id]);
        }

        Tensor &output(unsigned id)
        {
            if(id >= outputs_.size())
                throw ValidationError("Invalid output id");
            return tensor(outputs_[id]);
        }
        

    private:
        struct Connection {
            std::string name;
            std::unique_ptr<Operator> op;
            
            std::vector<std::string> parameter_names;
            std::vector<std::string> input_names;
            std::vector<Tensor>      input_tensors;
            std::vector<TensorSpecs> input_specs;

            std::vector<std::string> output_names;
            std::vector<Tensor>      output_tensors;
            std::vector<TensorSpecs> output_specs;
            
            std::vector<TensorSpecs> parameter_specs;
            std::vector<Tensor>      parameters;

            std::vector<TensorAndGradient> in_grad;
            std::vector<TensorAndGradient> out_grad;
            std::vector<TensorAndGradient> param_grad;
            size_t ws_size;
        };

        void setup_ws();
        void allocate_tensors();


        Context ctx_;
        std::map<std::string,TensorSpecs> tensor_specs_;
        std::map<std::string,TensorSpecs> parameter_specs_;

        std::vector<Connection> connections_;
        std::map<std::string,unsigned> connections_index_;

        Tensor workspace_;
        std::map<std::string,Tensor> tensors_;
        std::map<std::string,Tensor> tensors_diff_;

        std::map<std::string,Tensor> parameters_;
        std::map<std::string,Tensor> parameters_diff_;
        std::vector<std::string> inputs_;
        std::vector<std::string> outputs_;

        CalculationsMode mode_;
    };
};
