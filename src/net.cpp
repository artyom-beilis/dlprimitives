#include <dlprim/net.hpp>
#include <dlprim/json.hpp>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "H5Cpp.h"

namespace dlprim {
    Net::Net(Context &ctx) :
        ctx_(ctx)
    {
    }

    void Net::add_input_tensor(std::string const &name,TensorSpecs const &ts)
    {
        if(!tensor_specs_.insert(std::make_pair(name,ts)).second) {
            throw ValidationError("Tensor " + name + " already exists");
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
                Shape ds_shape(dims.begin(),dims.end());
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

    void Net::load_from_json(json::value const &v)
    {
        json::array const &inputs = v["inputs"].array();
        for(auto const &input:inputs) {
            auto const &vsp = input.get<std::vector<int> >("shape");
            Shape sp(vsp.begin(),vsp.end());
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
        conn.op->setup(conn.input_specs,conn.output_specs,conn.ws_size);
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
            OperatorWithParameters *pop = dynamic_cast<OperatorWithParameters *>(conn.op.get());
            if(!pop) {
                if(!parameters.empty()) {
                    throw ValidationError("Operator " + name + " does not have parameters. But names are provided");
                }
            }
            else {
                unsigned params_no = pop->parameter_specs().size();
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
                        parameter_specs_[pname] = pop->parameter_specs()[i];
                    }
                    else {
                        if(p->second != pop->parameter_specs()[i]) {
                            std::ostringstream ss;
                            ss << "Conflicting requirements for parameters specifications " << p->second << " vs " 
                               << pop->parameter_specs()[i] << " for " << pname;
                            throw ValidationError(ss.str());
                        }
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
        if(ws > 0)
            workspace_ = Tensor(ctx_,Shape(ws),uint8_data);
        else
            workspace_ = Tensor();
        for(auto &c : connections_) {
            if(c.ws_size > 0)
                c.op->set_workspace(workspace_);
        }
    }

    void Net::allocate_tensors()
    {
        tensors_.clear();
        parameters_.clear();
        for(auto const &ts : tensor_specs_) {
            tensors_[ts.first ] = Tensor(ctx_,ts.second.shape(),ts.second.dtype());
        }
        for(auto const &ps : parameter_specs_) {
            parameters_[ps.first ] = Tensor(ctx_,ps.second.shape(),ps.second.dtype());
        }
        for(auto &conn : connections_) {
            conn.input_tensors.clear();
            conn.output_tensors.clear();
            for(auto const &name : conn.input_names)
                conn.input_tensors.push_back(tensors_[name]);

            for(auto const &name : conn.output_names)
                conn.output_tensors.push_back(tensors_[name]);

            if(conn.parameter_names.empty())
                continue;
            OperatorWithParameters &pop = dynamic_cast<OperatorWithParameters&>(*conn.op);
            std::vector<Tensor> &tensors = pop.parameters();
            tensors.clear();
            for(auto const &name : conn.parameter_names) {
                tensors.push_back(parameters_[name]);
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


    void Net::reshape()
    {
        std::vector<Shape> in,out;
        for(auto &conn : connections_) {
            in.clear();
            out.clear();
            for(Tensor &s : conn.input_tensors)
                in.push_back(s.shape());
            conn.op->reshape(in,out);
            for(unsigned i=0;i<out.size();i++)
                conn.output_tensors[i].reshape(out[i]);
        }
    }

    void Net::clear_memory()
    {
        for(auto &conn : connections_) {
            conn.op->set_workspace();
            OperatorWithParameters *pop = dynamic_cast<OperatorWithParameters *>(conn.op.get());
            if(pop) {
                pop->parameters().clear();
            }
            conn.input_tensors.clear();
            conn.output_tensors.clear();
        }
        tensors_.clear();
        parameters_.clear();
    }

    void Net::forward(ExecutionContext const &e)
    {
        ExecGuard g(e,"forward");
        for(size_t i=0;i<connections_.size();i++) {
            ExecGuard g(e,connections_[i].name.c_str());
            ExecutionContext ec = e.generate_series_context(i,connections_.size());
            connections_[i].op->forward(connections_[i].input_tensors,connections_[i].output_tensors,ec);
        }
    }

      
} 
