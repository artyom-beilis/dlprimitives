#include <dlprim/net.hpp>
#include <dlprim/json.hpp>
#include <sstream>
#include <fstream>
#include <algorithm>

#ifndef DISABLE_HDF5
#include "H5Cpp.h"
#endif

namespace dlprim {
    Net::Net(Context &ctx) :
        ctx_(ctx),
        mode_(CalculationsMode::predict)
    {
    }

    void Net::mode(CalculationsMode m)
    {
        mode_ = m;
        for(auto &c:connections_)
            c.op->mode(m);
    }

    void Net::add_input_tensor(std::string const &name,TensorSpecs const &ts)
    {
        if(!tensor_specs_.insert(std::make_pair(name,ts)).second) {
            throw ValidationError("Tensor " + name + " already exists");
        }
        inputs_.push_back(name);
    }
    void Net::mark_output_tensor(std::string const &name)
    {
        if(tensor_specs_.find(name) == tensor_specs_.end()) {
            throw ValidationError("mark_output_tensor::No such tensor name " + name);
        }
        outputs_.push_back(name);
    }
#ifdef DISABLE_HDF5
    void Net::save_parameters_to_hdf5(std::string const &)
    {
        throw ValidationError("Library was build without HDF5 support");
    }
    void Net::load_parameters_from_hdf5(std::string const &,bool)
    {
        throw ValidationError("Library was build without HDF5 support");
    }
#else
    void Net::save_parameters_to_hdf5(std::string const &fname)
    {
        try {
            H5::H5File f(fname,H5F_ACC_TRUNC);
            for(auto  &pr : parameters_) {
                std::string name = pr.first;
                Tensor &tensor = pr.second;
                Shape shape = tensor.shape();
                std::vector<hsize_t> dims(1 + shape.size());
                for(int i=0;i<shape.size();i++)
                    dims[i] = shape[i];
                H5::DataSpace dsp(shape.size(),dims.data());
                H5::FloatType datatype( H5::PredType::NATIVE_FLOAT );
                datatype.setOrder( H5T_ORDER_LE );
                H5::DataSet dataset = f.createDataSet( name , datatype, dsp );
                if(tensor.dtype() == float_data) {
                    dataset.write(tensor.data<float>(),H5::PredType::NATIVE_FLOAT);
                }
                else {
                    throw ValidationError("FIXME load float16 from hdf5");
                }
            }
            f.close();
        }
        catch(H5::Exception const &e) {
            throw ValidationError("Failed to load HDF5 file " + fname + ": " + std::string(e.getCDetailMsg()));
        }

        
    }

    void Net::load_parameters_from_hdf5(std::string const &fname,bool allow_missing)
    {
        try {
            H5::H5File f(fname,H5F_ACC_RDONLY);
            for(auto  &pr : parameters_) {
                std::string name = pr.first;
                Tensor &tensor = pr.second;
                H5::DataSet dataset;
                try {
                    dataset = f.openDataSet(name);
                }
                catch(H5::Exception const &e) {
                    if(allow_missing)
                        continue;
                    throw;
                }
                H5::DataSpace dsp = dataset.getSpace();
                int ndims = dsp.getSimpleExtentNdims();
                std::vector<hsize_t> dims(ndims);
                dsp.getSimpleExtentDims(dims.data(),nullptr);
                Shape ds_shape=Shape::from_range(dims.begin(),dims.end());
                if(ds_shape != tensor.shape()) {
                    std::ostringstream ss;
                    ss << "Tensor shape mistmatch for " << name << " expecting " << tensor.shape() << " got " << ds_shape;
                    throw ValidationError(ss.str());
                }
                if(tensor.dtype() == float_data) {
                    dataset.read(tensor.data<float>(),H5::PredType::NATIVE_FLOAT);
                }
                else {
                    throw ValidationError("FIXME load float16 from hdf5");
                }
            }
        }
        catch(H5::Exception const &e) {
            throw ValidationError("Failed to load HDF5 file " + fname + ": " + std::string(e.getCDetailMsg()));
        }

        
    }
#endif
    void Net::load_from_json(json::value const &v)
    {
        json::array const &inputs = v["inputs"].array();
        for(auto const &input:inputs) {
            auto const &vsp = input.get<std::vector<int> >("shape");
            Shape sp=Shape::from_range(vsp.begin(),vsp.end());
            DataType dt(string_to_data_type(input.get("dtype","float")));
            TensorSpecs spec(sp,dt);
            std::string name = input.get("name","data");
            add_input_tensor(name,spec);
        }
        json::array const &operators = v["operators"].array();
        json::value empty_options = json::object();
        for(size_t i=0;i<operators.size();i++) {
            json::value const &op = operators[i];
            std::string name = op.get<std::string>("name");
            std::string type = op.get<std::string>("type");
            std::vector<std::string> inputs  = op.get("inputs", std::vector<std::string>());
            std::vector<std::string> outputs = op.get("outputs",std::vector<std::string>());
            std::vector<std::string> params  = op.get("params", std::vector<std::string>());
            json::value const &opts = op.find("options").is_undefined() ? empty_options : op["options"];
            std::unique_ptr<Operator> oper = create_by_name(ctx_,type,opts);
            add_operator(std::move(oper),name,inputs,outputs,params);
        }
        std::vector<std::string> outputs = v.get<std::vector<std::string> >("outputs");
        for(auto const &name : outputs) {
            mark_output_tensor(name);
        }
    }

    void Net::load_from_json_file(std::string const &name)
    {
        std::ifstream f(name);
        json::value net;
        int line=-1;
        if(!net.load(f,true,&line)) {
            throw ValidationError("Failed to load json from " + name + ", syntax error at line " + std::to_string(line));
        }
        load_from_json(net);
    }


    void Net::add_operator( std::unique_ptr<Operator> op,
                            std::string const &name,
                            std::vector<std::string> const &inputs,
                            std::vector<std::string> const &outputs,
                            std::vector<std::string> const &parameters)
    {
        Connection conn;
        conn.op = std::move(op);
        conn.op->mode(mode_);
        conn.name = name;
        if(connections_index_.find(name) != connections_index_.end()) {
            throw ValidationError("Operator with name " + name + " exists");
        }
        for(size_t i=0;i<inputs.size();i++) {
            auto spec_it = tensor_specs_.find(inputs[i]);
            if(spec_it == tensor_specs_.end()) {
                throw ValidationError("No such tensor " + inputs[i]);
            }
            conn.input_specs.push_back(spec_it->second);
        }
        conn.input_names = inputs;
        conn.op->setup(conn.input_specs,conn.output_specs,conn.parameter_specs,conn.ws_size);
        if(conn.output_specs.size() != outputs.size()) {
            throw ValidationError("Operator " + name + " expects to have " + std::to_string(conn.output_specs.size()) + 
                   " outputs, but only " + std::to_string(outputs.size()) + " provided");
        }
        conn.output_names = outputs;
        for(size_t i=0;i<outputs.size();i++) {
            auto p = tensor_specs_.find(outputs[i]);
            if(p == tensor_specs_.end()) {
                tensor_specs_[outputs[i]] = conn.output_specs[i];
            }
            else {
                if(p->second != conn.output_specs[i]) {
                    std::ostringstream ss;
                    ss << "Tensor " << outputs[i] << " is already defined with spec " << p->second << " but operator "
                       << name << " requires following output specs " << conn.output_specs[i];
                    throw ValidationError(ss.str());
                }
                if(std::find(inputs.begin(),inputs.end(),outputs[i]) == inputs.end()) {
                    throw ValidationError("Output " + outputs[i] + " for operator " + name + " aleady exists "
                            " howover it isn't as as input, output tensor can't have different sources other "
                            " then self/in-place operations ");
                }
            }
            unsigned params_no = conn.parameter_specs.size();
            if(params_no < parameters.size()) {
                std::ostringstream ss;
                ss << "Too many parameter names for operaror " << name << " expecting " << params_no << " got " << parameters.size();
                throw ValidationError(ss.str());
            }
            conn.parameter_names = parameters;
            conn.parameter_names.resize(params_no);
            for(size_t i=0;i<conn.parameter_names.size();i++) {
                std::string &pname = conn.parameter_names[i];
                if(pname.empty() || pname == "auto") {
                    conn.parameter_names[i] = name + "." + std::to_string(i);
                }
                auto p = parameter_specs_.find(pname);
                if(p==parameter_specs_.end()) {
                    parameter_specs_[pname] = conn.parameter_specs[i];
                }
                else {
                    if(p->second != conn.parameter_specs[i]) {
                        std::ostringstream ss;
                        ss << "Conflicting requirements for parameters specifications " << p->second << " vs " 
                           << conn.parameter_specs[i] << " for " << pname;
                        throw ValidationError(ss.str());
                    }
                }
            }
        }

        connections_index_[name] = connections_.size();
        connections_.push_back(std::move(conn));
    }

    void Net::setup()
    {
        clear_memory();
        setup_ws();
        allocate_tensors();
    }

    void Net::setup_ws()
    {
        size_t ws = 0;
        for(auto &c : connections_)
            ws = std::max(c.ws_size,ws);
        if(ws > 0) {
            if(workspace_.memory_size() < ws) {
                workspace_ = Tensor(); // clear first
                workspace_ = Tensor(ctx_,Shape(ws),uint8_data);
            }
        }
        else
            workspace_ = Tensor();
    }

    void Net::allocate_tensors()
    {
        tensors_.clear();
        parameters_.clear();
        bool train = mode_ == CalculationsMode::train;
        for(auto const &ts : tensor_specs_) {
            tensors_[ts.first ] = Tensor(ctx_,ts.second.shape(),ts.second.dtype());
            if(train)
                tensors_diff_[ts.first ] = Tensor(ctx_,ts.second.shape(),ts.second.dtype());
        }
        for(auto const &ps : parameter_specs_) {
            parameters_[ps.first ] = Tensor(ctx_,ps.second.shape(),ps.second.dtype());
            if(train)
                parameters_diff_[ps.first ] = Tensor(ctx_,ps.second.shape(),ps.second.dtype());
        }
        for(auto &conn : connections_) {
            conn.input_tensors.clear();
            conn.output_tensors.clear();
            for(auto const &name : conn.input_names) {
                conn.input_tensors.push_back(tensors_[name]);
                if(train) {
                    TensorAndGradient tg;
                    tg.data = tensors_[name];
                    tg.diff = tensors_diff_[name];
                    tg.accumulate_gradient = 0.0;
                    tg.requires_gradient = std::find(inputs_.begin(),inputs_.end(),name) == inputs_.end();
                    conn.in_grad.push_back(tg);
                }
            }

            for(auto const &name : conn.output_names) {
                conn.output_tensors.push_back(tensors_[name]);
                if(train) {
                    TensorAndGradient tg;
                    tg.data = tensors_[name];
                    tg.diff = tensors_diff_[name];
                    tg.accumulate_gradient = 0.0;
                    tg.requires_gradient = true;
                    conn.out_grad.push_back(tg);
                }
            }

            if(conn.parameter_names.empty())
                continue;
            conn.parameters.clear();
            for(auto const &name : conn.parameter_names) {
                conn.parameters.push_back(parameters_[name]);
                if(train) {
                    TensorAndGradient tg;
                    tg.data = parameters_[name];
                    tg.diff = parameters_diff_[name];
                    tg.accumulate_gradient = 1.0f;
                    tg.requires_gradient = true;
                    conn.param_grad.push_back(tg);
                }
            }
        }
    }

    void Net::copy_parameters_to_device()
    {
        cl::CommandQueue q = ctx_.make_queue();
        for(auto &pr : parameters_) {
            pr.second.to_device(q);
        }
    }
    void Net::copy_parameters_to_host()
    {
        cl::CommandQueue q = ctx_.make_queue();
        for(auto &pr : parameters_) {
            pr.second.to_host(q);
        }
    }


    void Net::reshape()
    {
        std::vector<Shape> in,out;
        bool train = mode_ == CalculationsMode::train;
        for(auto &conn : connections_) {
            in.clear();
            out.clear();
            for(Tensor &s : conn.input_tensors)
                in.push_back(s.shape());
            conn.op->reshape(in,out,conn.ws_size);
            for(unsigned i=0;i<out.size();i++) {
                conn.output_tensors[i].reshape(out[i]);
                if(train) {
                    conn.out_grad[i].diff.reshape(out[i]);
                }
            }
        }
        setup_ws();
    }

    void Net::clear_memory()
    {
        for(auto &conn : connections_) {
            conn.input_tensors.clear();
            conn.output_tensors.clear();
            conn.in_grad.clear();
            conn.out_grad.clear();
            conn.param_grad.clear();
        }
        workspace_ = Tensor();
        tensors_.clear();
        tensors_diff_.clear();
        parameters_.clear();
        parameters_diff_.clear();
    }

    void Net::forward(ExecutionContext const &e,bool sync)
    {
        ExecGuard g(e,"forward");
        for(size_t i=0;i<connections_.size();i++) {
            ExecGuard g(e,connections_[i].name.c_str());
            ExecutionContext ec = e.generate_series_context(i,connections_.size());
            connections_[i].op->forward(
                connections_[i].input_tensors,
                connections_[i].output_tensors,
                connections_[i].parameters,
                workspace_,
                ec);
            if(sync && ctx_.is_opencl_context())
                e.queue().finish();
        }
    }
    
    void Net::backward(ExecutionContext const &e,bool sync)
    {
        ExecGuard g(e,"backward");
        for(int i=connections_.size() - 1,it=0;i >= 0;i--,it++) {
            ExecGuard g(e,connections_[i].name.c_str());
            ExecutionContext ec = e.generate_series_context(it,connections_.size());
            connections_[i].op->backward(
                connections_[i].in_grad,
                connections_[i].out_grad,
                connections_[i].param_grad,
                workspace_,
                ec);
            if(sync && ctx_.is_opencl_context())
                e.queue().finish();
        }
    }
      
} 
