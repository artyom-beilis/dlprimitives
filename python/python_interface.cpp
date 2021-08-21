#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/net.hpp>
#include <dlprim/solvers/adam.hpp>
#include <dlprim/solvers/sgd.hpp>
#include <dlprim/json.hpp>
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
        TensorSpecs ns(nd_shape,np_dt);
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
}


BOOST_PYTHON_MODULE(_pydlprim)
{
    Py_Initialize();
    np::initialize();
    using namespace dlprim;

    bp::enum_<CalculationsModePy>("CalculationsMode").
        value("TRAIN",TRAIN).
        value("PREDICT",PREDICT).
        export_values();

    bp::enum_<DataType>("DataType").
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
   
    bp::class_<cl::CommandQueue>("CommandQueue");
 
    bp::class_<Context>("Context",bp::init<std::string>()).
        add_property("name",&Context::name). 
        add_property("is_opencl_context",&Context::is_opencl_context).
        add_property("is_cpu_context",&Context::is_cpu_context).
        def("make_execution_context",&Context::make_execution_context);


    bp::class_<ExecutionContext>("ExecutionContext").
        def(bp::init<>()).
        def(bp::init<cl::CommandQueue &>()).
        def("finish",&ExecutionContext::finish);
        

    bp::class_<Shape>("Shape").
        def(bp::init<>()).
        def(bp::init<size_t>()).
        def(bp::init<size_t,size_t>()).
        def(bp::init<size_t,size_t,size_t>()).
        def(bp::init<size_t,size_t,size_t,size_t>()).
        def("__len__",&Shape::size).
        def("__getitem__",&shape_getitem).
        add_property("total_size",&Shape::total_size).
        def("__str__",&shape_to_str);

    bp::class_<dp::Tensor>("Tensor").
        def(bp::init<>()).
        def(bp::init<Context &,Shape>()).
        def(bp::init<Context &,Shape,DataType>()).
        def("reshape",&Tensor::reshape).
        def("to_device",to_device).
        def("to_host",to_host).
        add_property("shape",dp::get_shape).
        add_property("dtype",&dp::Tensor::dtype);

    bp::class_<dp::Net,boost::noncopyable>("Net",bp::init<Context &>()).
        def("load_from_json_file",&Net::load_from_json_file).
        def("load_from_json",load_net_from_json).
        add_property("keep_intermediate_tensors",
            static_cast<bool (Net::*)() const>(&Net::keep_intermediate_tensors),
            static_cast<void (Net::*)(bool)>(&Net::keep_intermediate_tensors)).
        add_property("mode",net_mode_get,net_mode_set).
        def("setup",&Net::setup).
        def("initialize_parameters",&Net::initialize_parameters).
        def("reshape",&Net::reshape).
        def("forward",net_forward).
        def("backward",net_backward).
        def("tensor",&net_get_tensor).
        def("param",&net_get_param).
        add_property("input_names",&net_get_input_names).
        add_property("output_names",&net_get_output_names).
        def("load_parameters",
            static_cast<void (Net::*)(std::string const &,bool v)>(&Net::load_parameters)).
        def("load_parameters",net_load_parameters).
        def("save_parameters",&Net::save_parameters).
        def("save_parameters_to_hdf5",&Net::save_parameters_to_hdf5);

    using solvers::Adam;
    bp::class_<Adam>("Adam",bp::init<Context &>()).
        def_readwrite("lr",&Adam::lr).
        def_readwrite("beta1",&Adam::beta1).
        def_readwrite("beta2",&Adam::beta2).
        def_readwrite("weight_decay",&Adam::weight_decay).
        def_readwrite("eps",&Adam::eps).
        def("init",&Adam::init).
        def("zero_grad",&Adam::zero_grad).
        def("apply",&Adam::apply).
        def("step",&Adam::step);

    using solvers::SGD;
    bp::class_<SGD>("SGD",bp::init<Context &>()).
        def_readwrite("lr",&SGD::lr).
        def_readwrite("momentum",&SGD::momentum).
        def_readwrite("weight_decay",&SGD::weight_decay).
        def("init",&SGD::init).
        def("zero_grad",&SGD::zero_grad).
        def("apply",&SGD::apply).
        def("step",&SGD::step);


}
