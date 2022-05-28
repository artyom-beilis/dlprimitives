///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/operator.hpp>
#include <map>
#include <vector>
#include <list>

namespace dlprim {
    class SharedResource;
    class ModelBase;

    ///
    /// Major object used for inference
    ///
    class Net {
    public:

        ///
        /// Create an empty network object for a context
        ///
        Net(Context &ctx);

        ~Net() {}
        Net const &operator=(Net const &) = delete;
        Net(Net const &) = delete;
        
        Net &operator=(Net &&) = default;
        Net(Net &&) = default;

        ///
        /// Load from external model, shortcut to load_from_json(mode.network()) + setup() + load_parameters(model)
        ///
        void load_model(ModelBase &model);
        ///
        /// Configure network graph from json
        ///
        void load_from_json(json::value const &v);
        ///
        /// Configure network graph from json
        ///
        void load_from_json_file(std::string const &name);

        ///
        /// Define network input
        ///
        void add_input_tensor(std::string const &name,TensorSpecs const &ts,bool requires_gradient=false);
        ///
        /// Make a tensor as an output tensor (will be preserverd) by name
        ///
        void mark_output_tensor(std::string const &name);
        ///
        /// Make a tensor as an output loss tensor, regardless of its name (by default a name starting with "loss")
        /// considered loss
        ///
        void set_loss_weight(std::string const &name,float loss_weight=1.0);

        ///
        /// Load parameters from binary stream DLP format (file) s must be seekable
        ///
        void load_parameters(std::istream &s,bool allow_missing=false);
        ///
        /// Load parameters from file name can be either DLP or HDF5 format
        ///
        void load_parameters(std::string const &name,bool allow_missing=false);
        ///
        /// Load parameters from external model
        ///
        void load_parameters(ModelBase &model,bool allow_missing=false);

        ///
        /// Save network parameters to DLP format
        ///
        void save_parameters(std::string const &fname);

        ///
        /// Copy all parameters for hdf5 file format. allow_missing - ignore if parameter is not defined
        ///
        void load_parameters_from_hdf5(std::string const &fname,bool allow_missing=false);
        ///
        /// Save all network parameters to a file
        ///
        void save_parameters_to_hdf5(std::string const &fname);

        ///
        /// True if all intermediate results are kept
        /// 
        bool keep_intermediate_tensors() const
        {
            return keep_intermediate_tensors_;
        }
        ///
        /// Set if to keed intermediate results for debugging. Default is false - optimise memory use
        /// and reuse intermediate memory chunks
        ///
        void keep_intermediate_tensors(bool keep) 
        {
            keep_intermediate_tensors_ = keep;
        }

        ///
        /// Add an operator \a op to the network. name should be unique
        ///
        /// inputs - tensor input names - must be defined (as input or output of another operatos)
        /// outputs - tensor output names
        /// parameters - give cutom names to parameters
        /// 
        void add_operator(  std::unique_ptr<Operator> op,
                            std::string const &name,
                            std::vector<std::string> const &inputs,
                            std::vector<std::string> const &outputs,
                            std::vector<std::string> const &parameters = std::vector<std::string>(),
                            bool frozen=false);

        /// Set the calculations mode
        ///
        /// default is testing. If setup was called when the network in testing mode it can't be switched to training
        /// if you plan to use the network for training call it _before_ setup()
        void mode(CalculationsMode mode); 

        /// get currenct calcuations mode
        CalculationsMode mode() const
        {
            return mode_;
        }

        ///
        /// Get SharedResource object that can be shared between all operators in same network allowing operator 
        /// access to network level features
        ///
        std::shared_ptr<SharedResource> shared_resource()
        {
            return shared_resource_;
        }

        ///
        /// Allocate all tensors and prepare network data
        ///
        void setup();
        ///
        /// reshape all tensors in network after reshaping the input data
        ///
        void reshape();

        ///
        /// Initialize all parameters for training
        ///
        void initialize_parameters(ExecutionContext const &e);

        void copy_parameters_to_device(); 
        void copy_parameters_to_host(); 
        void clear_memory();

        ///
        /// Get all intermediate tensors (operators inputs/outputs). Note if keep_intermediate_tensors() == false that same
        /// memory may be shared between different tensors
        ///
        std::map<std::string,Tensor> &tensors()
        {
            return tensors_;
        }
        ///
        /// Get all intermediate tensor (operators inputs/outputs) gradient. Note if keep_intermediate_tensors() == false that same
        /// memory may be shared between different tensors
        ///
        std::map<std::string,Tensor> &tensor_diffs()
        {
            return tensors_diff_;
        }

        ///
        /// All operator parameters trainable and not trainable
        ///
        std::map<std::string,Tensor> &params()
        {
            return parameters_;
        }
        ///
        /// All operator parameters gradients trainable and not trainable
        ///
        std::map<std::string,Tensor> &param_diffs()
        {
            return parameters_diff_;
        }

        /// Get tensor by name, throws ValidationError if does not exist
        Tensor &tensor(std::string const &name)
        {
            auto p=tensors_.find(name);
            if(p == tensors_.end())
                throw ValidationError("Unknown tensor name:" + name);
            return p->second;
        }
        /// Get tensor gradient by name, throws ValidationError if does not exist
        Tensor &tensor_diff(std::string const &name)
        {
            auto p=tensors_diff_.find(name);
            if(p == tensors_diff_.end())
                throw ValidationError("Unknown tensor name:" + name);
            return p->second;
        }
        
        /// Get parameter by name, throws ValidationError if does not exist
        Tensor &param(std::string const &name)
        {
            auto p=parameters_.find(name);
            if(p == parameters_.end())
                throw ValidationError("Unknown parameter name:" + name);
            return p->second;
        }
        /// Get parameter gradient by name, throws ValidationError if does not exist
        Tensor &param_diff(std::string const &name)
        {
            auto p=parameters_diff_.find(name);
            if(p == parameters_diff_.end())
                throw ValidationError("Unknown parameter name:" + name);
            return p->second;
        }

        void forward(ExecutionContext const &ectx,bool sync=false);
        void backward(ExecutionContext const &ectx,bool sync=false);

        /// Get of all network inputs
        std::vector<std::string> const &input_names()
        {
            return inputs_;
        }

        /// Get of all network outputs
        std::vector<std::string> const &output_names()
        {
            return outputs_;
        }

        /// Get input tensors by their numeric id - according to the order in input_names()
        Tensor &input(unsigned id)
        {
            if(id >= inputs_.size())
                throw ValidationError("Invalid input id");
            return tensor(inputs_[id]);
        }

        /// Get output tensors by their numeric id - according to the order in output_names()
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
            int gradient_flags;
            bool frozen;
        };

        void setup_ws();
        void mark_backpropagating_edges();
        void allocate_tensors();
        void allocate_aliases();
        bool is_loss(std::string const &name);
        void allocate_optimized_chunks(bool forward_only);
        void tensor_use_list(std::vector<std::list<std::string> > &start,
                              std::vector<std::list<std::string> > &stop);
        void allocate_chunks();
        void load_header(std::istream &f,json::value &v);


        Context ctx_;
        std::map<std::string,TensorSpecs> tensor_specs_;
        std::map<std::string,TensorSpecs> parameter_specs_;

        std::vector<Connection> connections_;
        std::map<std::string,unsigned> connections_index_;

        Tensor workspace_;
        std::map<std::string,std::string> alias_sources_;
        std::map<std::string,Tensor> tensors_;
        std::map<std::string,Tensor> tensors_diff_;

        std::map<std::string,Tensor> parameters_;
        std::map<std::string,Tensor> parameters_diff_;
        std::map<std::string,float> loss_weights_;
        std::vector<std::string> inputs_;
        std::vector<std::string> outputs_;
        std::vector<Tensor> memory_;

        std::shared_ptr<SharedResource> shared_resource_;

        CalculationsMode mode_;
        bool keep_intermediate_tensors_;
    };
};
