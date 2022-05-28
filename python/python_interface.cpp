///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/net.hpp>
#include <dlprim/solvers/adam.hpp>
#include <dlprim/solvers/sgd.hpp>
#include <dlprim/json.hpp>

#ifdef WITH_ONNX
#include <dlprim/onnx.hpp>
#endif

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <string>
#include <sstream>
namespace bp = boost::python;
namespace np = boost::python::numpy;
namespace dp = dlprim;

namespace dlprim {
    static bp::list vec_to_plist(std::vector<std::string> const &v)
    {
        bp::list l;
        for(std::string const &s : v) {
            l.append(s);
        }
        return l;
    }

    static bp::dict map_shape(std::map<std::string,dp::Shape> const &s)
    {
        bp::dict r;
        for(auto const &pair : s) {
            std::vector<int> shape(pair.second.begin(),pair.second.end());
            bp::list l;
            for(int i:shape)
                l.append(i);
            r[pair.first] = l;
        }
        return r;
    }

    static bp::list net_get_input_names(Net &n)
    {
        return vec_to_plist(n.input_names());
    }
    static bp::list net_get_output_names(Net &n)
    {
        return vec_to_plist(n.output_names());
    }

    static void compare_specs(Tensor &t,np::ndarray const &ar)
    {
        int n = ar.get_nd();
        Shape nd_shape = Shape::from_range(ar.get_shape(),ar.get_shape()+n);
        std::string np_type = bp::extract<std::string>(bp::str(ar.get_dtype()));
        DataType np_dt = string_to_data_type(np_type);
        TensorSpecs ns(nd_shape,np_dt,t.is_trainable());
        if(t.specs() != ns) {
            std::ostringstream msg;
            msg << "numpy array and tensor have different shapes/types tensor="  << t.specs() << " ndarray=" << ns; 
            throw ValidationError(msg.str());
        }
        if(ar.get_dtype().get_itemsize() != size_of_data_type(t.dtype()))
            throw ValidationError("numpy and dlprim data type size differ for " + data_type_to_string(t.dtype()));
        if(!(ar.get_flags() & np::ndarray::C_CONTIGUOUS))
            throw ValidationError("numpy array must be contiguous, call np.ascontiguousarray first");
    }
    void to_device(Tensor &t,np::ndarray const &ar,ExecutionContext const &e)
    {
        compare_specs(t,ar);
        t.to_device(e,ar.get_data(),true);
    }
    void to_host(Tensor &t,np::ndarray const &ar,ExecutionContext const &e)
    {
        compare_specs(t,ar);
        t.to_host(e,ar.get_data(),true);
    }

    static Shape get_shape(Tensor const &t)
    {
        return t.shape();
    }
    static std::string shape_to_str(Shape const &s)
    {
        std::ostringstream ss;
        ss<<s;
        return ss.str();
    }
    static size_t shape_getitem(Shape const &s,int i)
    {
        if(i < 0 || i>= s.size()) {
            std::ostringstream ss;
            ss << "Invalid shape index " << i << " for shape " << s;
            throw ValidationError(ss.str());
        }
        return s[i];
    }
    static void load_net_from_json(Net &net,std::string const &s)
    {
        json::value v;
        char const *begin = s.c_str();
        char const *end = begin + s.size();
        int line=-1;
        if(!v.load(begin,end,true,&line)) {
            throw ValidationError("Invalid Json, parsing failed in line " + std::to_string(line));
        }
        net.load_from_json(v);
    }

    Tensor net_get_tensor(Net &n,std::string const &name)
    {
        return n.tensor(name);
    }
    Tensor net_get_param(Net &n,std::string const &name)
    {
        return n.param(name);
    }

    void net_forward(Net &n,ExecutionContext const &e)
    {
        n.forward(e);
    }
    void net_backward(Net &n,ExecutionContext const &e)
    {
        n.backward(e);
    }
    void net_load_parameters(Net &n,std::string const &path)
    {
        n.load_parameters(path);
    }

    enum CalculationsModePy {
        TRAIN, PREDICT
    };

    int net_mode_get(Net &n)
    {
        if(n.mode() == CalculationsMode::train)
            return TRAIN;
        else
            return PREDICT;
    }
    void net_mode_set(Net &n,int m)
    {
        if(m==TRAIN)
            n.mode(CalculationsMode::train);
        else
            n.mode(CalculationsMode::predict);
    }
#ifdef WITH_ONNX
    std::string onnx_net(ONNXModel &model)
    {
        std::ostringstream ss;
        return model.network().save(dlprim::json::readable);
    }
    bp::dict onnx_input_shapes(ONNXModel &model)
    {
        return map_shape(model.input_shapes());
    }
    bp::dict onnx_dynamic_axes(ONNXModel &model)
    {
        bp::dict r;
        for(auto const &p : model.dynamic_axes()) {
            bp::list axes;
            for(auto const &axis : p.second)
                axes.append(axis);
            r[p.first] = axes;
        }
        return r;
    }
#endif
}


BOOST_PYTHON_MODULE(_pydlprim)
{
    Py_Initialize();
    np::initialize();
    using namespace dlprim;

    bp::enum_<CalculationsModePy>("CalculationsMode","Train or Preduct mode selection").
        value("TRAIN",TRAIN).
        value("PREDICT",PREDICT).
        export_values();

    bp::enum_<DataType>("DataType","Type of tensor data").
        value("float32",float_data).
        value("half",half_data).
        value("bfloat16",bfloat16_data).
        value("uint8",uint8_data).
        value("int8",int8_data).
        value("uint16",uint16_data).
        value("int16",int16_data).
        value("int32",int32_data).
        value("uint32",uint32_data).
        value("int64",int64_data).
        value("uint64",uint64_data).
        export_values();
   
    bp::class_<cl::CommandQueue>("CommandQueue","Wrapper of cl::CommandQueue");

 
    bp::class_<Context>("Context","Context of computations - represents OpenCL device, platform and context, it is passed to all object"
                "that are created during the process").
        def(bp::init<std::string>("Create context by name, which is either `cpu` or P:D where P is platform id (number) and D is device id (number)")).
        add_property("name",&Context::name,"Get name of the context - device name and platform readable to human"). 
        add_property("is_opencl_context",&Context::is_opencl_context,"returns true if the context is OpenCL context").
        add_property("is_cpu_context",&Context::is_cpu_context,"returns true if context is CPU Context").
        def("make_execution_context",&Context::make_execution_context,"Creates execution context - wrapper of OpenCL command queue");


    bp::class_<ExecutionContext>("ExecutionContext","Execution context - represents OpenCL command queue").
        def(bp::init<>("Create empty queue - for CPU")).
        def(bp::init<cl::CommandQueue &>("Create command queue from OpenCL object")).
        def("finish",&ExecutionContext::finish,"Synchronize queue, once it completes all operations are finished");
        

    bp::class_<Shape>("Shape","Shape of tensor object").
        def(bp::init<>()).
        def(bp::init<size_t>()).
        def(bp::init<size_t,size_t>()).
        def(bp::init<size_t,size_t,size_t>()).
        def(bp::init<size_t,size_t,size_t,size_t>()).
        def("__len__",&Shape::size,"get number of dimentions of the shape").
        def("__getitem__",&shape_getitem,"get size for a dimention").
        add_property("total_size",&Shape::total_size,"get total count of items in shape, for example for Shape(2,3) it is 6").
        def("__str__",&shape_to_str,"format as string");

    bp::class_<dp::Tensor>("Tensor","Class that represents a tensor").
        def(bp::init<>("Create Null tensor")).
        def(bp::init<Context &,Shape>("Create float32 tensor for context of given shape")).
        def(bp::init<Context &,Shape,DataType>("Create a tensor for context of given shape and type")).
        def("reshape",&Tensor::reshape,"Change shape of the tensor - its total_size of new shape needs to be <= of current total size").
        def("to_device",to_device,"Enqueue copy tensor to defice from a numpy array and wait for completion. numpy array must have same shape, type as tensor and must be contiguous, operation is always synchronous").
        def("to_host",to_host,"Enqueue copy tensor to host to a numpy array and wait for completion. numpy array must have same shape, type as tensor and must be contiguous, operation is always synchronous").
        add_property("shape",dp::get_shape,"Returns tensor shape").
        add_property("dtype",&dp::Tensor::dtype,"Returns tensor type");
    
    bp::class_<dp::ModelBase>("ModelBase",bp::no_init);
#ifdef WITH_ONNX
    bp::class_<dp::ONNXModel,bp::bases<dp::ModelBase>,boost::noncopyable >("ONNXModel",
                "ONNX Model parser and importer, once it is loaded, Net.load can be called").
        def(bp::init<>("Empty")).
        add_property("network",&onnx_net,"Export network as string").
        add_property("input_shapes",&onnx_input_shapes,"input shapes - to update in needed before loading").
        add_property("dynamic_axes",&onnx_dynamic_axes,"list of dynamic axes per input, for example {'data':[0]}").
        def("set_dynamic_axis",&dp::ONNXModel::set_dynamic_axis,"Modify dynamic shape - set maximal limit an axis for input, if the shape isn't fixed rises a error").
        def("set_batch",&dp::ONNXModel::set_batch,"Shortcut to update first dim of all inputs").
        def("load",&dp::ONNXModel::load,"Load model from file").
        def("build",&dp::ONNXModel::build,"Build network");

#endif

    bp::class_<dp::Net,boost::noncopyable>("Net",
            R"xxxxx(Central class that represents network for training and inference
    General workflow for network preparaiton is following:

        ctx = dlprim.Context(device_name) # general context
        exe_context = ctx.make_execution_context() # create execution queue

        net = dlprim.Net(ctx) # create network for context
        net.mode = dlprim.TRAIN # set mode to training
        net.load_from_json(net_config)  # configure graph
        net.setup() # allocate all tensors for graph
        net.initialize_parameters(exe_context) # initialize parameters

        Than the main training loop is:
        opt = dlprim.Adam(ctx) # create optilized
        opt.init(net,exe_context) # initialize it

        dev_data = net.tensor("data")
        dev_labals = net.tensor("labels")
        dev_loss = net.tensor("loss")
        for each batch:
            dev_data.to_device(my_data,exe_context)
            dev_loss.to_device(my_labels,exe_context)
            opt.step(net,exe_context)
            # asynchronous execution started
            # now we can prepara next batch
            my_data = my_next_data
            my_labels = my_next_labels
            loss.to_host(my_loss,exe_context) 
            # ^ sync op - so we wait for completion
            print(my_loss)
            
    
    For inference it is following:        

        net = dlprim.Net(ctx) # create network for context
        net.load_from_json_file(path_net_config_js)  # configure graph
        net.setup() # allocate all tensors for graph
        net.load_parameters(path_to_weights) # initialize parameters


        # coyp input data
        net.tensors("data").to_device(my_data,exe_context)
        net.forward(exe_context) # start computing - async
        net.tensors(my_prob,exe_context) # get results

    Network that created in training mode can switch to inference most
    and back by setting mode propery
        )xxxxx"
            ,bp::init<Context &>("Create a network for a context")).
        def("load_from_json_file",&Net::load_from_json_file,"Load network definition from json file").
        def("load_from_json",load_net_from_json,"Load network definition from json text" ).
        add_property("keep_intermediate_tensors",
            static_cast<bool (Net::*)() const>(&Net::keep_intermediate_tensors),
            static_cast<void (Net::*)(bool)>(&Net::keep_intermediate_tensors),
            "set to true to keed intermediate results for debugging. Default is false - optimise memory use and reuse intermediate memory chunks").
        add_property("mode",net_mode_get,net_mode_set,
            "Property for changing network behavior, by default network is created for PREDICT mode for inference, if you want to train network, set to TRAIN before calling setup\n"
            "note: network created for inference can't be switched to train mode, but network created for training can work in both PREDICT and TRAIN mode").
        def("setup",&Net::setup,"Allocate all tensors and prepare network data").
        def("initialize_parameters",&Net::initialize_parameters,"Enqueue network parameters initialization with default values for training").
        def("reshape",&Net::reshape,"reshape all tensors in network after reshaping the input data").
        def("forward",net_forward,"Enqueue network forward propogation to the execution context and return").
        def("backward",net_backward,"Enqueue network backpropogation to the execution context and return").
        def("tensor",&net_get_tensor,"Get tensor by name").
        def("param",&net_get_param,"Get network parameter by name").
        add_property("input_names",&net_get_input_names,"List of all input tensor names").
        add_property("output_names",&net_get_output_names,"List of all output tensor names").
        def("load_parameters",
            static_cast<void (Net::*)(std::string const &,bool v)>(&Net::load_parameters),
            "Load parameters from file, either HDF5 or DLP format. If second argument - ignore missing parameter, if true and parameter value does not exists in file it isn't considered a error, useful for transfer learning").
        def("load_parameters",net_load_parameters,"same as self.load_parameters(file,False)").
        def("save_parameters",&Net::save_parameters,"save nework parameters to DLP format").
        def("save_parameters_to_hdf5",&Net::save_parameters_to_hdf5,"save nework parameters to HDF5 format").
        def("load_model",&Net::load_model,"Load external model");

    using solvers::Adam;
    bp::class_<Adam>("Adam","Adam optimizer",bp::init<Context &>("Create optimizer from context")).
        def_readwrite("lr",&Adam::lr,"Lear rate, default 0.001").
        def_readwrite("beta1",&Adam::beta1,"beta1, default 0.9").
        def_readwrite("beta2",&Adam::beta2,"beta2, default 0.999").
        def_readwrite("weight_decay",&Adam::weight_decay,"weight decay, default 0.0005").
        def_readwrite("eps",&Adam::eps,"epsilon, default  1e-8").
        def("init",&Adam::init,"Prapare optimizer for network with execution_context, asynchronous").
        def("zero_grad",&Adam::zero_grad,"zero all network gradients before accumulation").
        def("apply",&Adam::apply,"apply optimizer step - update all weights according to its status/gradients").
        def("step",&Adam::step,R"xx(shortcut to 
        zero_grad(net,exe_ctx) 
        net.forward(exe_ctx)
        net.backward(exe_ctx)
        apply(net,exe_ctx)
        )xx"
        );

    using solvers::SGD;
    bp::class_<SGD>("SGD","SGD optimizer",bp::init<Context &>("Create SDG for context")).
        def_readwrite("lr",&SGD::lr,"learn rate, default 0.1").
        def_readwrite("momentum",&SGD::momentum,"momenutum, default 0.9").
        def_readwrite("weight_decay",&SGD::weight_decay,"weight decay, default 0.0005").
        def("init",&SGD::init,"Prapare optimizer for network with execution_context, asynchronous").
        def("zero_grad",&SGD::zero_grad,"zero all network gradients before accumulation").
        def("apply",&SGD::apply,"apply optimizer step - update all weights according to its status/gradients").
        def("step",&SGD::step,R"xx(shortcut to 
        zero_grad(net,exe_ctx) 
        net.forward(exe_ctx)
        net.backward(exe_ctx)
        apply(net,exe_ctx)
        )xx"
        );


}
