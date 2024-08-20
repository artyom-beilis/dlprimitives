///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/net.hpp>
#include <dlprim/json.hpp>
#include <dlprim/shared_resource.hpp>
#include <dlprim/ops/initialization.hpp>
#include <dlprim/model.hpp>
#include <sstream>
#include <fstream>
#include <set>
#include <list>
#include <algorithm>

#ifndef DISABLE_HDF5
#include "H5Cpp.h"
#endif

namespace dlprim {
    Net::Net(Context &ctx) :
        ctx_(ctx),
        shared_resource_(new SharedResource()),
        mode_(CalculationsMode::predict),
        keep_intermediate_tensors_(false)
    {
    }

    void Net::mode(CalculationsMode m)
    {
        mode_ = m;
        for(auto &c:connections_)
            c.op->mode(m);
    }

    void Net::add_input_tensor(std::string const &name,TensorSpecs const &ts,bool requires_gradient)
    {
        TensorSpecs new_ts(ts.shape(),ts.dtype(),requires_gradient);
        if(!tensor_specs_.insert(std::make_pair(name,new_ts)).second) {
            throw ValidationError("Tensor " + name + " already exists");
        }
        inputs_.push_back(name);
    }
    void Net::set_loss_weight(std::string const &name,float lw)
    {
        mark_output_tensor(name);
        loss_weights_[name] = lw;
    }
    void Net::mark_output_tensor(std::string const &name)
    {
        if(tensor_specs_.find(name) == tensor_specs_.end()) {
            throw ValidationError("mark_output_tensor::No such tensor name " + name);
        }
        if(std::find(outputs_.begin(),outputs_.end(),name)==outputs_.end()) {
            outputs_.push_back(name);
            if(name.find("loss")==0)
                loss_weights_[name] = 1.0f;
        }
    }
    void Net::save_parameters(std::string const &fname)
    {
        json::value header;
        json::value &tensors = header["tensors"];
        tensors = json::object();
        size_t start_pos = 0;
        for(auto  &pr : parameters_) {
            std::string name = pr.first;
            json::value &tensor_specs = tensors[name];
            tensor_specs["dtype"] = data_type_to_string(pr.second.dtype());
            Shape shape = pr.second.shape();
            for(int i=0;i<shape.size();i++)
                tensor_specs["shape"][i] = shape[i];
            size_t mem = pr.second.memory_size();
            tensor_specs["size"] = mem;
            tensor_specs["start"] = start_pos;
            start_pos += mem;
        }
        std::ostringstream ss;
        ss << header;
        std::string header_content = ss.str();
        unsigned len = header_content.size();
        std::ofstream f(fname,std::fstream::binary);
        f<< "DLPW";
        for(int i=0;i<4;i++) {
            unsigned char v = 0xFF & (len >> (24 - i*8));
            f << (char)(v);
        }
        f << header_content;
        for(auto  &pr : parameters_) {
            void *ptr = pr.second.host_data();
            size_t len = pr.second.memory_size();
            f.write((char*)ptr,len);
        }
        f.flush();
        if(!f) {
            throw ValidationError("I/O error in saving to " + fname);
        }
    }

    void Net::load_model(ModelBase &model)
    {
        load_from_json(model.network());
        setup();
        load_parameters(model);
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
        copy_parameters_to_device();
    }
#endif
    void Net::initialize_parameters(ExecutionContext const &e)
    {
        for(auto &conn : connections_) {
            conn.op->initialize_params(conn.parameters,e);
        }
        if(mode() == CalculationsMode::train) {
            // set loss diff
            for(auto const &name : output_names()) {
                if(is_loss(name))
                    set_to_constant(tensor_diff(name),loss_weights_[name],e);
            }
        }
    }
    void Net::load_header(std::istream &f,json::value &v)
    {
        unsigned char buf[8]={0};
        f.read((char*)buf,8);
        if(memcmp(buf,"DLPW",4) != 0)
            throw ValidationError("Invalid File Format");
        unsigned len = 0;
        for(int i=0;i<4;i++) {
            len |= unsigned(buf[4+i]) << ((3-i)*8);
        }
        if(len > 1024*1024*64)
            throw ValidationError("Header seems to be too big");
        std::vector<char> buffer(len+1);
        f.read(buffer.data(),len);
        if(!f)
            throw ValidationError("Problem readfing file");
        buffer[len] = 0;
        char const *begin=&buffer[0];
        char const *end  =&buffer[len]; // (there is +1)
        if(!v.load(begin,end,true)) {
            throw ValidationError("Problem parsing content");
        }
    }
    void Net::load_parameters(std::string const &file_name,bool allow_missing)
    {
        std::ifstream f(file_name,std::ifstream::binary);
        char buf[5]={};
        f.read(buf,4);
        if(buf==std::string("DLPW")) {
            f.seekg(0);
            try {
                load_parameters(f,allow_missing);
            }
            catch(std::exception const &e) {
                throw ValidationError(std::string(e.what()) + " in file " + file_name);
            }
        }
        else if(buf==std::string("\211HDF")) {
            f.close();
            load_parameters_from_hdf5(file_name,allow_missing);
        }
        else {
            throw ValidationError("Unidentified majic number for " + file_name);
        }
    }

    void Net::load_parameters(ModelBase &model,bool allow_missing)
    {
        for(auto  &pr : parameters_) {
            std::string name = pr.first;
            Tensor &tensor = pr.second;
            Tensor value = model.get_parameter(name);
            if(value.shape().size() == 0) {
                if(allow_missing)
                    continue;
                throw ValidationError("No parameter " + name + " was found");
            }
            if(tensor.shape() != value.shape() || tensor.dtype() != value.dtype()) {
                std::ostringstream err;
                err << "Expected " << tensor << " parameter, got " << value << " for " << name;
                throw ValidationError(err.str());
            }
            memcpy(tensor.host_data(),value.host_data(),tensor.memory_size());
        }
        copy_parameters_to_device();
    }
    void Net::load_parameters(std::istream &f,bool allow_missing)
    {
        json::value v;
        load_header(f,v);
        size_t offset = f.tellg();
        json::object const &tensors=v.find("tensors").object();
        for(auto  &pr : parameters_) {
            std::string name = pr.first;
            Tensor &tensor = pr.second;
            auto p = tensors.find(name);
            if(p == tensors.end()) {
                if(allow_missing)
                    continue;
                throw ValidationError("No parameter " + name + " was found");
            }
            std::vector<int> dims = p->second.get<std::vector<int> >("shape");
            DataType dt = string_to_data_type(p->second.get<std::string>("dtype"));
            Shape ds_shape=Shape::from_range(dims.begin(),dims.end());
            if(ds_shape != tensor.shape() || dt != tensor.dtype()) {
                std::ostringstream ss;
                ss << "Tensor shape/type mistmatch for " << name << " expecting " << tensor << " got " << ds_shape << " " << data_type_to_string(dt);
                throw ValidationError(ss.str());
            }
            size_t start = p->second.get<size_t>("start");
            size_t size  = p->second.get<size_t>("size");
            if(size != tensor.memory_size()) {
                throw ValidationError("Object size mistmatch");
            }
            f.seekg(start + offset);
            f.read(static_cast<char *>(tensor.host_data()),size);
            if(!f) {
                throw ValidationError("I/O error");
            }
        }
        copy_parameters_to_device();
    }
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
            bool frozen = op.get<bool>("frozen",false);
            std::vector<std::string> inputs  = op.get("inputs", std::vector<std::string>());
            std::vector<std::string> outputs = op.get("outputs",std::vector<std::string>());
            std::vector<std::string> params  = op.get("params", std::vector<std::string>());
            json::value const &opts = op.find("options").is_undefined() ? empty_options : op["options"];
            std::unique_ptr<Operator> oper = create_by_name(ctx_,type,opts);
            add_operator(std::move(oper),name,inputs,outputs,params,frozen);
        }
        for(json::value const &output : v["outputs"].array()) {
            if(output.type() == json::is_string)
                mark_output_tensor(output.str());
            else {
                std::string const &name = output.get<std::string>("name");
                if(output.find("loss_weight").is_undefined())
                    mark_output_tensor(name);
                else
                    set_loss_weight(name,output.get<float>("loss_weight"));
            }
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
                            std::vector<std::string> const &parameters,
                            bool frozen)
    {
        Connection conn;
        conn.op = std::move(op);
        conn.op->shared_resource(shared_resource());
        conn.op->mode(mode_);
        conn.frozen = frozen;
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
            if(conn.op->alias_generator()) {
                if(inputs.size() != outputs.size()) {
                    throw ValidationError("Inputs need to have same size for operator " + name);
                }
                for(size_t i=0;i<inputs.size();i++) {
                    if(conn.input_specs[i].dtype() != conn.output_specs[i].dtype()
                       || conn.input_specs[i].shape().total_size() != conn.output_specs[i].shape().total_size())
                    {
                        throw ValidationError("Alias operator need to have tensors of same type and size, only shape may be altered for "+ name);
                    }
                    std::string src = inputs[i];
                    std::map<std::string,std::string>::iterator p;
                    while((p=alias_sources_.find(src))!=alias_sources_.end())
                        src = p->second;
                    alias_sources_[outputs[i]]=src;
                }
            }
        }

        if(frozen) {
            for(auto &spec : conn.parameter_specs) {
                spec.freeze();
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

        connections_index_[name] = connections_.size();
        connections_.push_back(std::move(conn));
    }

    void Net::setup()
    {
        clear_memory();
        mark_backpropagating_edges();
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

    bool Net::is_loss(std::string const &name)
    {
        return loss_weights_.find(name)!=loss_weights_.end();
    }

    void Net::mark_backpropagating_edges()
    {
        std::map<std::string,int> requires_gradient;
        for(std::string const &name : inputs_) {
            if(tensor_specs_[name].is_trainable())
                requires_gradient[name] |= 1;
        }
        for(std::string const &name : outputs_) {
            if(is_loss(name))
                requires_gradient[name] |= 2;
        }
        for(size_t i=0;i<connections_.size();i++) {
            connections_[i].gradient_flags = 0;
        }
        // needs gradient
        for(size_t i=0;i<connections_.size();i++) {
            int &gradient_flags = connections_[i].gradient_flags;
            for(auto const &spec : connections_[i].parameter_specs) {
                if(spec.is_trainable()) {
                    gradient_flags |= 1;
                    break;
                }
            }
            
            for(auto const &name : connections_[i].input_names) {
                if(requires_gradient[name] & 1) {
                    gradient_flags |= 1;
                    break;
                }
            }
            if((gradient_flags & 1) == 0)
                continue;
            for(auto const &name : connections_[i].output_names) {
                requires_gradient[name] |= 1;
            }
        }
        // needs gradient
        for(int i=connections_.size()-1;i>=0;i--) {
            int &gradient_flags = connections_[i].gradient_flags;
            for(auto const &name : connections_[i].output_names) {
                if(requires_gradient[name] & 2) {
                    gradient_flags |= 2;
                    break;
                }
            }
            if((connections_[i].gradient_flags & 2) == 0)
                continue;
            for(auto const &name : connections_[i].input_names) {
                requires_gradient[name] |= 2;
            }
        }
        for(auto &ts : tensor_specs_) {
            if(requires_gradient[ts.first]!=3) {
                ts.second.freeze();
            }
        }
    }

    void Net::tensor_use_list(std::vector<std::list<std::string> > &start,
                              std::vector<std::list<std::string> > &stop)
    {
        std::map<std::string,std::pair<int,int> > used_at;
        int last = connections_.size()-1;
        for(auto const &n : inputs_) 
            used_at[n] = std::make_pair(0,last);
        for(auto const &n : outputs_) 
            used_at[n] = std::make_pair(0,last);
        for(int i=0;i<int(connections_.size());i++) {
            for(int dir=0;dir<2;dir++) {
                for(auto const &tensor_name : (dir == 0 ? connections_[i].input_names : connections_[i].output_names)) {
                    std::string name;
                    auto p = alias_sources_.find(tensor_name);
                    if(p != alias_sources_.end())
                        name = p->second;
                    else
                        name = tensor_name;

                    if(used_at.find(name) == used_at.end()) {
                        used_at[name] = std::make_pair(i,i);
                    }
                    else {
                        auto &fl = used_at[name];
                        fl.first  = std::min(fl.first,i);
                        fl.second = std::max(fl.second,i);
                    }
                }
            }
        }
        start.clear();
        start.resize(connections_.size());
        stop.clear();
        stop.resize(connections_.size());
        for(auto const &v:used_at) {
            start[v.second.first].push_back(v.first);
            stop[v.second.second].push_back(v.first);
        }
    }

    void Net::allocate_optimized_chunks(bool forward)
    {
        std::vector<std::list<std::string> > alloc_needed;
        std::vector<std::list<std::string> > free_needed;
        tensor_use_list(alloc_needed,free_needed);
        if(!forward)
            alloc_needed.swap(free_needed);

        std::multimap<size_t,int> pool;
        std::vector<size_t> chunks;
        std::map<std::string,int> chunks_mapping;
        int step_offset,step_scale;
        if(forward) {
            step_offset = 0;
            step_scale = 1;
        }
        else {
            step_offset = connections_.size() - 1;
            step_scale = -1;
        }
        for(unsigned i=0;i<connections_.size();i++) {
            int index = i * step_scale + step_offset;
            for(auto const &n : alloc_needed[index]) {
                size_t mem = tensor_specs_[n].memory_size();
                if(pool.empty()) {
                    chunks_mapping[n] = chunks.size();
                    chunks.push_back(mem);
                }
                else {
                    auto p = pool.lower_bound(mem);
                    if(p!=pool.end()) {
                        chunks_mapping[n] = p->second;
                        pool.erase(p);
                    }
                    else {
                        auto last = pool.rbegin();
                        chunks_mapping[n] = last->second;
                        chunks[last->second] = mem; // increase memory
                        pool.erase(std::prev(pool.end()));
                    }
                }
            }
            for(auto const &n : free_needed[index]) {
                int cid = chunks_mapping[n];
                pool.insert(std::make_pair(chunks[cid],cid));
            }
        }
        memory_.clear();
        for(size_t i=0;i<chunks.size();i++) {
            memory_.push_back(Tensor(ctx_,Shape(chunks[i]),uint8_data));
        }
        tensors_.clear();
        tensors_diff_.clear();
        for(auto const &ts : tensor_specs_) {
            if(alias_sources_.count(ts.first) > 0)
                continue;
            int cid = chunks_mapping[ts.first];
            auto base_tensor = memory_[cid];
            Tensor actual_tensor = base_tensor.sub_tensor(0,ts.second.shape(),ts.second.dtype());
            if(forward) {
                tensors_[ts.first ] = actual_tensor;
            }
            else {
                tensors_[ts.first ] = Tensor(ctx_,ts.second.shape(),ts.second.dtype());
                tensors_diff_[ts.first ] = actual_tensor;
            }
        }
        allocate_aliases();

    }
    void Net::allocate_chunks()
    {
        tensors_.clear();
        tensors_diff_.clear();
        memory_.clear();
        bool train = mode_ == CalculationsMode::train;
        // normal
        for(auto const &ts : tensor_specs_) {
            if(alias_sources_.count(ts.first) != 0)
                continue;
            tensors_[ts.first ] = Tensor(ctx_,ts.second.shape(),ts.second.dtype());
            if(train)
                tensors_diff_[ts.first ] = Tensor(ctx_,ts.second.shape(),ts.second.dtype());
        }
        allocate_aliases();
    }

    void Net::allocate_aliases()
    {
        bool train = mode_ == CalculationsMode::train;
        // alias
        for(auto const &ts : tensor_specs_) {
            auto p = alias_sources_.find(ts.first);
            if(p==alias_sources_.end())
                continue;
            std::string src_name = p->second;
            tensors_[ts.first ] = tensors_[src_name].alias(ts.second.shape());
            if(train)
                tensors_diff_[ts.first ] = tensors_diff_[src_name].alias(ts.second.shape());
        }
    }

    void Net::allocate_tensors()
    {
        tensors_.clear();
        tensors_diff_.clear();
        parameters_.clear();
        bool train = mode_ == CalculationsMode::train;
        if(keep_intermediate_tensors_)
            allocate_chunks();
        else
            allocate_optimized_chunks(!train); // forward = !train
        for(auto const &ps : parameter_specs_) {
            parameters_[ps.first ] = Tensor(ctx_,ps.second.shape(),ps.second.dtype(),ps.second.is_trainable());
            if(train && ps.second.is_trainable())
                parameters_diff_[ps.first ] = Tensor(ctx_,ps.second.shape(),ps.second.dtype());
        }

        /// FWD connections
        for(auto &conn : connections_) {
            conn.input_tensors.clear();
            conn.output_tensors.clear();
            for(auto const &name : conn.input_names) {
                conn.input_tensors.push_back(tensors_[name]);
            }

            for(auto const &name : conn.output_names) {
                conn.output_tensors.push_back(tensors_[name]);
            }

            if(conn.parameter_names.empty())
                continue;

            conn.parameters.clear();
            for(auto const &name : conn.parameter_names) {
                conn.parameters.push_back(parameters_[name]);
            }
        }

        /// BWD Connections
        std::set<std::string> zeroed_grad;
        for(int index=connections_.size()-1;index>=0;index--) {
            auto &conn = connections_[index];
            conn.in_grad.clear();
            conn.out_grad.clear();
            conn.param_grad.clear();
            bool is_trainable = train && conn.gradient_flags == 3;
            for(auto const &name : conn.input_names) {
                bool in_place = std::find(conn.output_names.begin(),
                                          conn.output_names.end(),
                                          name) != conn.output_names.end();
                TensorAndGradient tg;
                tg.data = tensors_[name];
                if(tensor_specs_[name].is_trainable() && is_trainable) {
                    tg.diff = tensors_diff_[name];
                    if(zeroed_grad.find(name) == zeroed_grad.end() || in_place) {
                        tg.accumulate_gradient = 0.0; // first in BP zeroes gradient
                        zeroed_grad.insert(name);
                    }
                    else {
                        tg.accumulate_gradient = 1.0; // next accumulate
                    }
                    tg.requires_gradient = true; 
                }
                else
                    tg.requires_gradient = false;
                conn.in_grad.push_back(tg);
            }

            for(auto const &name : conn.output_names) {
                TensorAndGradient tg;
                tg.data = tensors_[name];
                if(is_trainable)
                    tg.diff = tensors_diff_[name];
                tg.accumulate_gradient = 0.0;
                tg.requires_gradient = true;
                conn.out_grad.push_back(tg);
            }

            if(conn.parameter_names.empty())
                continue;
            for(auto const &name : conn.parameter_names) {
                TensorAndGradient tg;
                tg.data = parameters_[name];
                if(tg.data.is_trainable() && is_trainable) {
                    tg.diff = parameters_diff_[name];
                    tg.accumulate_gradient = 1.0f;
                    tg.requires_gradient = true;
                }
                else {
                    tg.accumulate_gradient = 0;
                    tg.requires_gradient = false;
                }
                conn.param_grad.push_back(tg);
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
                if(train && tensor_specs_[conn.output_names[i]].is_trainable()) {
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
        memory_.clear();
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
            if(connections_[i].gradient_flags != 3)
                continue;
            connections_[i].op->backward(
                connections_[i].in_grad,
                connections_[i].out_grad,
                connections_[i].param_grad,
                workspace_,
                ec);
            if(sync && ctx_.is_opencl_context()) {
                e.queue().finish();
            }

        }
    }
      
} 
