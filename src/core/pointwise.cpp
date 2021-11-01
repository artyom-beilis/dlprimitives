#include <dlprim/core/common.hpp>
#include <dlprim/gpu/program_cache.hpp>

#include <iostream>
#include <sstream>

namespace dlprim {
namespace core {

    void bind_as_dtype(cl::Kernel &k,int &p,double value,DataType dt)
    {
        switch(dt) {
        case  double_data:  k.setArg(p++,double(value)); break;
        case  float_data:   k.setArg(p++,float(value)); break;
        case  int64_data:   k.setArg(p++,cl_long(value)); break;
        case  int32_data:   k.setArg(p++,cl_int(value)); break;
        case  int16_data:   k.setArg(p++,cl_short(value)); break;
        case  int8_data:    k.setArg(p++,cl_char(value)); break;
        case  uint64_data:  k.setArg(p++,cl_long(value)); break;
        case  uint32_data:  k.setArg(p++,cl_int(value)); break;
        case  uint8_data:   k.setArg(p++,cl_uchar(value)); break;
        default:
            throw  NotImplementedError("Unsupported bind as type:" + data_type_to_opencl_type(dt));
        }
    }

    void pointwise_operation(std::vector<Tensor> xs,
                             std::vector<Tensor> ys,
                             std::vector<double>  ws,
                             std::string const &code,
                             ExecutionContext const &e)
    {
        Context ctx(e);
        Shape ref;
        DataType ref_type = float_data;
        DLPRIM_CHECK(xs.size() + ys.size() > 0);
        if(xs.empty()) {
            ref = ys[0].shape();
            ref_type = ys[0].dtype();
        }
        else {
            ref = xs[0].shape();
            ref_type = xs[0].dtype();
        }

        for(size_t i=0;i<xs.size();i++) {
            DLPRIM_CHECK(ref == xs[i].shape());
            DLPRIM_CHECK(ref_type == xs[i].dtype());
        }
        for(size_t i=0;i<ys.size();i++) {
            DLPRIM_CHECK(ref == ys[i].shape());
            DLPRIM_CHECK(ref_type == ys[i].dtype());
        }
        std::ostringstream params,loads,saves;
        for(size_t i=0;i<xs.size();i++) {
            params<<", __global dtype const *px" << i<< ", ulong px"<<i<<"_offset ";
            loads<<"dtype x"<<i<<"=px"<<i<<"[index + px"<<i<<"_offset]; ";
        }
        for(size_t i=0;i<ys.size();i++) {
            params<<", __global dtype *py" << i<< ", ulong py"<<i<<"_offset ";
            loads<<"dtype y"<<i<<";";
            saves<<"py"<<i<<"[index]=y"<<i<<"; ";
        }
        for(size_t i=0;i<ws.size();i++) {
            params<<", dtype w" <<i;
        }

        std::ostringstream code_fixed;
        for(size_t i=0;i<code.size();i++)
            if(code[i]=='\n')
                code_fixed << "\\\n";
            else
                code_fixed << code[i];
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"pointwise",
                                                                           "dtype",data_type_to_opencl_type(ref_type),
                                                                           "#PARAMS",params.str(),
                                                                           "#LOADS",loads.str(),
                                                                           "#SAVES",saves.str(),
                                                                           "#CALC",code_fixed.str());
        cl::Kernel k(prog,"exec");
        cl_ulong total = ref.total_size();
        int p=0;
        k.setArg(p++,total);
        for(Tensor &x:xs)
            x.set_arg(k,p);
        for(Tensor &y:ys)
            y.set_arg(k,p);
        for(double w:ws)
            bind_as_dtype(k,p,w,ref_type);
        e.queue().enqueueNDRangeKernel(k,cl::NullRange,cl::NDRange(total),cl::NullRange,e.events(),e.event("pointwise_exec"));
    }




    template<int size>
    struct __attribute__ ((packed)) CLShape {
        cl_ulong s[size];
    };

    template<int size>
    void bind_cl_shape(cl::Kernel &k,int &p,Shape const &s)
    {
        CLShape<size> cl_s;
        for(int i=0;i<size;i++)
            cl_s.s[i]=s[i];
        k.setArg(p++,cl_s);
    }

    void bind_shape(cl::Kernel &k,int &p,Shape const &s)
    {
        switch(s.size()) {
        case 1: bind_cl_shape<1>(k,p,s); return;
        case 2: bind_cl_shape<2>(k,p,s); return;
        case 3: bind_cl_shape<3>(k,p,s); return;
        case 4: bind_cl_shape<4>(k,p,s); return;
        case 5: bind_cl_shape<5>(k,p,s); return;
        default:
            {
                std::ostringstream ss;
                ss << "Shape isn't valid " << s;
                throw ValidationError(ss.str());
            }
        }
    }
    std::string format_code(std::string const &code)
    {
        std::ostringstream code_fixed;
        for(size_t i=0;i<code.size();i++)
            if(code[i]=='\n')
                code_fixed << "\\\n";
            else
                code_fixed << code[i];
        code_fixed << '\n';
        return code_fixed.str();
    }


    void pointwise_operation_broadcast( std::vector<Tensor> xs,
                                        std::vector<Tensor> ys,
                                        std::vector<double>  ws,
                                        std::string const &code,
                                        ExecutionContext const &e)
    {
        DLPRIM_CHECK(!xs.empty());
        DLPRIM_CHECK(!ys.empty());

        std::vector<Shape> shapes(xs.size() + ys.size());
        for(size_t i=0;i<xs.size();i++)
            shapes[i] = xs[i].shape();
        for(size_t j=0;j<ys.size();j++)
            shapes[j+xs.size()] = ys[j].shape();

        shrink_broadcast_ranges(shapes);

        DataType target_type = ys[0].dtype();
        Context ctx(e);
        Shape ref = shapes[xs.size()]; // ys[0]
        for(size_t i=0;i<ys.size();i++) {
            DLPRIM_CHECK(shapes[i + xs.size()] == ref);
        }

        std::vector<Shape> strides(xs.size());
        for(size_t i=0;i<xs.size();i++) {
            strides[i] = shapes[i].broadcast_strides(ref);
        }

        std::ostringstream params,loads,saves;
        for(size_t i=0;i<xs.size();i++) {
            std::string type = data_type_to_opencl_type(xs[i].dtype());
            params<<", __global " << type << " const *px" << i<< ", ulong px"<<i<<"_offset, Shape strides" << i;
            loads<<type << " x"<<i<<"=px"<<i<<"[get_offset(index,strides" << i << ",px"<<i<<"_offset)];\\\n";
        }
        for(size_t i=0;i<ys.size();i++) {
            std::string type = data_type_to_opencl_type(ys[i].dtype());
            params<<", __global "<<type << " *py" << i<< ", ulong py"<<i<<"_offset";
            loads<<type << " y"<<i<<";\\\n";
            saves<<"py"<<i<<"[get_direct_offset(index,limit,py"<<i<<"_offset)]=y"<<i<<";\\\n";
        }
        loads << "typedef " << data_type_to_opencl_type(target_type) <<  " target_type;\\\n";

        for(size_t i=0;i<ws.size();i++) {
            params<<", "<<data_type_to_opencl_type(target_type) << " w" <<i;
        }

        loads << '\n';
        saves <<'\n';
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,  "pointwise_broadcast",
                                                                           "DIMS",ref.size(),
                                                                           "#PARAMS",params.str(),
                                                                           "#LOADS",loads.str(),
                                                                           "#SAVES",saves.str(),
                                                                           "#CALC",format_code(code));
        cl::Kernel k(prog,"exec");
        int p=0;
        bind_shape(k,p,ref);
        for(size_t i=0;i<xs.size();i++) {
            xs[i].set_arg(k,p);
            bind_shape(k,p,strides[i]);
        }
        for(Tensor &y:ys)
            y.set_arg(k,p);
        
        for(double w:ws) 
            bind_as_dtype(k,p,w,target_type);
        cl::NDRange range;
        switch(ref.size()) {
        case 1: range = cl::NDRange(ref[0]); break;
        case 2: range = cl::NDRange(ref[1],ref[0]); break;
        case 3: range = cl::NDRange(ref[2],ref[1],ref[0]); break;
        case 4: range = cl::NDRange(ref[3]*ref[2],ref[1],ref[0]); break;
        case 5: range = cl::NDRange(ref[4]*ref[3],ref[2]*ref[1],ref[0]); break;
        default:
            throw NotImplementedError("Invalid dimentsions count for broadcastes shape size " + std::to_string(ref.size()));
        }

            
        e.queue().enqueueNDRangeKernel(k,cl::NullRange,range,cl::NullRange,e.events(),e.event("pointwise_exec_broadcast"));
    }


    void pointwise_operation_broadcast_reduce(  std::vector<Tensor> xs,
                                                std::vector<Tensor> ys,
                                                std::vector<double>  ws,
                                                std::string const &compute,
                                                std::string const &reduce_init,
                                                std::string const &reduce,
                                                ExecutionContext const &e)
    {
        DLPRIM_CHECK(!xs.empty());
        DLPRIM_CHECK(!ys.empty());

        std::vector<Shape> shapes(xs.size() + ys.size());
        for(size_t i=0;i<xs.size();i++)
            shapes[i] = xs[i].shape();
        for(size_t j=0;j<ys.size();j++)
            shapes[j+xs.size()] = ys[j].shape();

        shrink_broadcast_ranges(shapes);
        
        Shape ref = shapes[0]; // ys[0]
        for(size_t i=1;i<shapes.size();i++) {
            ref = broadcast(ref,shapes[i]);
        }
        // all yes same
        for(size_t i=xs.size()+1;i<shapes.size();i++) {
            DLPRIM_CHECK(shapes[xs.size()] == shapes[i]);
        }

        DataType target_type = ys[0].dtype();
        Context ctx(e);

        std::vector<Shape> strides(xs.size() + ys.size());
        for(size_t i=0;i<shapes.size();i++) {
            strides[i] = shapes[i].broadcast_strides(ref);
        }
        
        std::vector<int> reduce_dims,non_reduce_dims;
        {
            Shape ref_stride = strides[xs.size()];
            for(int dim = 0;dim < ref.size();dim++) {
                if(ref_stride[dim] == 0)
                    reduce_dims.push_back(dim);
                else
                    non_reduce_dims.push_back(dim);
            }
        }

        DLPRIM_CHECK(!reduce_dims.empty());

        for(size_t i=0;i<shapes.size() + 1;i++) {
            Shape &src = i < shapes.size() ? strides[i] : ref;
            Shape tgt = src;
            for(int dim=0;dim<ref.size();dim++) {
                int pos = 0;
                for(auto indx : reduce_dims)
                    tgt[pos++] = src[indx];
                for(auto indx : non_reduce_dims)
                    tgt[pos++] = src[indx];
            }
            src = tgt;
        }


        // all the defines
        std::ostringstream PARAMS,PREPARE_LOAD_INPUT_ALL,REDUCE_INIT_ALL,LOAD_INPUT_ALL,
            LOAD_REDUCE_ALL,SAVE_REDUCE_ALL,LOAD_REDUCED_SAVE_GLOBAL_ALL;
        
        for(size_t i=0;i<xs.size();i++) {
            std::string type = data_type_to_opencl_type(xs[i].dtype());
            std::string suffix = "(" + type + "," + std::to_string(i) + ") ";
            PARAMS << "PARAM_INPUT" << suffix;
            PREPARE_LOAD_INPUT_ALL << "PREPARE_LOAD_INPUT" << suffix << ";\\\n";
            LOAD_INPUT_ALL << "LOAD_INPUT(" << i << ");\\\n";
        }

        for(size_t i=0;i<ys.size();i++) {
            std::string type = data_type_to_opencl_type(ys[i].dtype());
            std::string suffix = "(" + type + "," + std::to_string(i) + ") ";
            PARAMS << "PARAM_OUTPUT" << suffix;
            REDUCE_INIT_ALL << "REDUCE_INIT"<<suffix << ";\\\n";
            LOAD_REDUCE_ALL << "LOAD_REDUCE("<<i<<");\\\n";
            SAVE_REDUCE_ALL << "SAVE_REDUCE("<<i<<");\\\n";
            LOAD_REDUCED_SAVE_GLOBAL_ALL << "LOAD_REDUCED_SAVE_GLOBAL("<<i<<");\\\n";
        }

        REDUCE_INIT_ALL << format_code(reduce_init) << "\n";

        for(size_t i=0;i<ws.size();i++) {
            PARAMS<<", "<<data_type_to_opencl_type(target_type) << " w" <<i;
        }

        size_t total_reduce = 1;
        for(unsigned i=0;i<reduce_dims.size();i++)
            total_reduce *= ref[i];

        int wg_size;
        if(total_reduce >= 256) {
            wg_size = 256;
        }
        else if(total_reduce >= 128) {
            wg_size = 128;
        }
        else {
            wg_size = 64;
        }
        int items_per_wi = (total_reduce + wg_size - 1) / wg_size;
        std::cerr << "Items per thread:" << items_per_wi << std::endl;
        int mpl = wg_size * items_per_wi;
        int nd_range = (total_reduce + mpl - 1) / mpl * wg_size;
        
        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,  "pointwise_broadcast_reduce",
                                                                           "REDUCE_DIMS",reduce_dims.size(),
                                                                           "DIMS",ref.size(),
                                                                           "WG_SIZE",wg_size,
                                                                           "ITEMS_PER_WI",items_per_wi,
                                                                           "#PARAMS",PARAMS.str(),
                                                                           "#PREPARE_LOAD_INPUT_ALL",PREPARE_LOAD_INPUT_ALL.str(),
                                                                           "#REDUCE_INIT_ALL",REDUCE_INIT_ALL.str(),
                                                                           "#LOAD_INPUT_ALL",LOAD_INPUT_ALL.str(),
                                                                           "#LOAD_REDUCE_ALL",LOAD_REDUCE_ALL.str(),
                                                                           "#SAVE_REDUCE_ALL",SAVE_REDUCE_ALL.str(),
                                                                           "#LOAD_REDUCED_SAVE_GLOBAL_ALL",LOAD_REDUCED_SAVE_GLOBAL_ALL.str(),
                                                                           "#REDUCE",format_code(reduce),
                                                                           "#CALC",format_code(compute));
        cl::Kernel k(prog,"exec");
        int p=0;
        bind_shape(k,p,ref);
        int stride_id = 0;
        for(Tensor &x:xs) {
            x.set_arg(k,p);
            bind_shape(k,p,strides[stride_id++]);
        }
        
        for(Tensor &y:ys) {
            y.set_arg(k,p);
            bind_shape(k,p,strides[stride_id++]);
        }
        
        for(double w:ws) {
            bind_as_dtype(k,p,w,target_type);
        }


        cl::NDRange range;
        int zero = reduce_dims.size();
        switch(non_reduce_dims.size()) {
        case 0: range = cl::NDRange(nd_range,1,                      1                      ); break;
        case 1: range = cl::NDRange(nd_range,ref[zero+0],            1                      ); break;
        case 2: range = cl::NDRange(nd_range,ref[zero+1],            ref[zero+0]            ); break;
        case 3: range = cl::NDRange(nd_range,ref[zero+2],            ref[zero+1]*ref[zero+0]); break;
        case 4: range = cl::NDRange(nd_range,ref[zero+3]*ref[zero+2],ref[zero+1]*ref[zero+0]); break;
        default:
            throw NotImplementedError("Invalid dimentsions count for broadcastes shape size " + std::to_string(ref.size()));
        }

        e.queue().enqueueNDRangeKernel(k,cl::NullRange,range,cl::NDRange(wg_size,1,1),e.events(),e.event("pointwise_exec_broadcast"));
    }


} // core
} // dlprim

