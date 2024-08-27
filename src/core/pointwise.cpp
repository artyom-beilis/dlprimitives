///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/common.hpp>
#include <dlprim/core/pointwise.hpp>
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
        case  half_data:    k.setArg(p++,float(value)); break; // half goes as float to kernel parameter
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

        std::string param_dtype = data_type_to_opencl_param_type(ref_type);

        for(size_t i=0;i<ws.size();i++) {
            params<<", " << param_dtype << " w" <<i;
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
    struct CLShape {
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
        case 6: bind_cl_shape<6>(k,p,s); return;
        case 7: bind_cl_shape<7>(k,p,s); return;
        case 8: bind_cl_shape<8>(k,p,s); return;
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


    cl::NDRange get_broadcast_ndrange(Shape ref)
    {
        cl::NDRange range;
        switch(ref.size()) {
        case 1: range = cl::NDRange(ref[0]); break;
        case 2: range = cl::NDRange(ref[1],ref[0]); break;
        case 3: range = cl::NDRange(ref[2],ref[1],ref[0]); break;
        case 4: range = cl::NDRange(ref[3]*ref[2],ref[1],ref[0]); break;
        case 5: range = cl::NDRange(ref[4]*ref[3],ref[2]*ref[1],ref[0]); break;
        case 6: range = cl::NDRange(ref[5]*ref[4],ref[3]*ref[2],ref[1]*ref[0]); break;
        case 7: range = cl::NDRange(ref[6]*ref[5]*ref[4],ref[3]*ref[2],ref[1]*ref[0]); break;
        case 8: range = cl::NDRange(ref[7]*ref[6]*ref[5],ref[4]*ref[3]*ref[2],ref[1]*ref[0]); break;
        default:
            throw NotImplementedError("Invalid dimentsions count for broadcastes shape size " + std::to_string(ref.size()));
        }
        return range;
    }

    cl::NDRange get_broadcast_reduce_ndrange(Shape ref,int zero,int non_reduce_dims,size_t nd_range)
    {
        cl::NDRange range;
        switch(non_reduce_dims) {
        case 0: range = cl::NDRange(nd_range,1,                       1                      ); break;
        case 1: range = cl::NDRange(nd_range,ref[zero+0],             1                      ); break;
        case 2: range = cl::NDRange(nd_range,ref[zero+1],             ref[zero+0]            ); break;
        case 3: range = cl::NDRange(nd_range,ref[zero+2],             ref[zero+1]*ref[zero+0]); break;
        case 4: range = cl::NDRange(nd_range,ref[zero+3]*ref[zero+2], ref[zero+1]*ref[zero+0]); break;
        default:
            throw NotImplementedError("Invalid dimentsions count for broadcastes shape size " + std::to_string(ref.size()));
        }
        return range;
    }

    void pointwise_operation_broadcast( std::vector<Tensor> xs,
                                        std::vector<Tensor> ys,
                                        std::vector<double> ws,
                                        std::string const &code,
                                        ExecutionContext const &e)
    {
        std::vector<DataType> dts(ws.size(),ys.at(0).dtype());
        pointwise_operation_broadcast(xs,ys,ws,dts,code,e);
    }

    void pointwise_operation_broadcast( std::vector<Tensor> xs,
                                        std::vector<Tensor> ys,
                                        std::vector<double> ws,
                                        std::vector<DataType> dts,
                                        std::string const &code,
                                        ExecutionContext const &e,
                                        bool shrink_dims)
    {
        DLPRIM_CHECK(!xs.empty());
        DLPRIM_CHECK(!ys.empty());
        DLPRIM_CHECK(ws.size() == dts.size());

        std::vector<Shape> shapes(xs.size() + ys.size());
        for(size_t i=0;i<xs.size();i++)
            shapes[i] = xs[i].shape();
        for(size_t j=0;j<ys.size();j++)
            shapes[j+xs.size()] = ys[j].shape();

        if(shrink_dims)
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
            loads<<"typedef " << type << " typeof_x" << i << ";\\\n";
        }
        for(size_t i=0;i<ys.size();i++) {
            std::string type = data_type_to_opencl_type(ys[i].dtype());
            params<<", __global "<<type << " *py" << i<< ", ulong py"<<i<<"_offset";
            loads<<type << " y"<<i<<";\\\n";
            saves<<"py"<<i<<"[get_direct_offset(index,limit,py"<<i<<"_offset)]=y"<<i<<";\\\n";
            loads<<"typedef " << type << " typeof_y" << i << ";\\\n";
        }
        loads << "typedef " << data_type_to_opencl_type(target_type) <<  " target_type;\\\n";

        for(size_t i=0;i<ws.size();i++) {
            std::string type = data_type_to_opencl_param_type(dts[i]);
            params<<", "<<type<< " w" <<i;
            loads<<"typedef " << type << " typeof_w" << i << ";\\\n";
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
        
        for(size_t i=0;i<ws.size();i++) { 
            bind_as_dtype(k,p,ws[i],dts[i]);
        }
        cl::NDRange range = get_broadcast_ndrange(ref);
            
        e.queue().enqueueNDRangeKernel(k,cl::NullRange,range,cl::NullRange,e.events(),e.event("pointwise_exec_broadcast"));
    }

    ///
    /// Perform pointwise operation with both boradcasting and reduction
    ///
    /// Calculation is performed over a shape that xs and ys tensors are boradcaasted to.
    ///
    /// For example xs have shapes: (64,10,5) and (64,10,1) and ys has shape (10,1) they all
    /// broadcast to 64,10,5 and reduction is performed over dimentsions 0 and 2
    ///
    /// All ys tensors need to have same shape and be boradcastable to total shape
    ///
    /// Optional parameters can be provided that avalible in code as w0... wN, Final ys are computed as `ys[i] = alpha[i] * reduced_result + beta[i] * ys[i]`
    ///
    class PointwiseOperationBroadcastReduceImpl : public PointwiseOperationBroadcastReduce {
    public:
        
        virtual ~PointwiseOperationBroadcastReduceImpl() {}
        ///
        /// Get size of workspace in bytes needed
        ///
        virtual size_t workspace() 
        {
            return ws_size_;
        }
        ///
        /// Perform coputations
        ///
        /// \param xs - vector of input tensor
        /// \param ys - vector of output tenors
        //  \param parameters - the weight paramerters, size should match weights_count
        /// \param alpha - scale for ys, must match size of ys
        /// \param beta - scale for summation of previous ys, must match size of ys
        ///
        ///
        virtual void enqueue(std::vector<Tensor> xs,
                             std::vector<Tensor> ys,
                             Tensor &workspace,
                             std::vector<double> parameters,
                             std::vector<double> alpha,
                             std::vector<double> beta,
                             ExecutionContext const &e)
        {
            DLPRIM_CHECK(xs.size() == xs_specs_.size());
            DLPRIM_CHECK(ys.size() == ys_specs_.size());
            DLPRIM_CHECK(ws_size_ == 0 || workspace.memory_size() >= ws_size_);
            DLPRIM_CHECK(parameters.size() == params_count_);

            for(size_t i=0;i<xs.size();i++) {
                DLPRIM_CHECK(xs[i].specs() == xs_specs_[i]);
            }
            for(size_t i=0;i<ys.size();i++) {
                DLPRIM_CHECK(ys[i].specs() == ys_specs_[i]);
            }
            int p=0;
            
            bind_shape(kernel_,p,ref_);
            int stride_id = 0;
            for(Tensor &x:xs) {
                x.set_arg(kernel_,p);
                bind_shape(kernel_,p,strides_[stride_id++]);
            }
           
            std::vector<Tensor> temp_ys; 
            std::vector<Tensor> temp_ys_outputs;
            for(size_t i=0;i<ys.size();i++) {
                if(second_stage_stride_ == 1) {
                    Tensor &y=ys[i];
                    y.set_arg(kernel_,p);
                    bind_shape(kernel_,p,strides_[stride_id++]);
                    bind_as_dtype(kernel_,p,alpha.at(i),ys[i].dtype());
                    bind_as_dtype(kernel_,p,beta.at(i),ys[i].dtype());
                }
                else {
                    Tensor temp_y = workspace
                        .workspace_as_type(uint8_data)
                        .sub_tensor(ws_offsets_[i].first,Shape(ws_offsets_[i].second),uint8_data)
                        .workspace_as_type(ys[i].dtype());
                    temp_y.reshape(Shape(ys[i].shape().total_size(),second_stage_stride_));
                    temp_ys.push_back(temp_y);
                    Tensor temp_yout = ys[i].sub_tensor(0,Shape(ys[i].shape().total_size(),1),ys[i].dtype());
                    temp_ys_outputs.push_back(temp_yout);
                    temp_y.set_arg(kernel_,p);
                    bind_shape(kernel_,p,strides_[stride_id++]);
                }
            }
            
            for(double w: parameters) {
                bind_as_dtype(kernel_,p,w,target_type_);
            }

            if(second_stage_stride_ != 1) {
                kernel_.setArg(p++,cl_ulong(second_stage_stride_));
            }

            if(second_stage_stride_ == 1) {
                e.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,range_,wg_range_,e.events(),e.event("pointwise_exec_broadcast"));
            }
            else {
                auto e1 = e.generate_series_context(0,2);
                auto e2 = e.generate_series_context(1,2);
                e.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,range_,wg_range_,e1.events(),e1.event("pointwise_exec_broadcast_1st_stage"));
                DLPRIM_CHECK(second_stage_->workspace() == 0);
                Tensor tmp;
                second_stage_->enqueue(temp_ys,temp_ys_outputs,tmp,{},alpha,beta,e2);
            }
        }

        PointwiseOperationBroadcastReduceImpl(  Context &ctx,
                                                std::vector<TensorSpecs> xs,
                                                std::vector<TensorSpecs> ys,
                                                int weights_count,DataType weights_type,
                                                std::string const &compute_code,
                                                std::string const &reduce_init,
                                                std::string const &reduce)
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

            target_type_ = weights_type;
            params_count_ = weights_count;
            ws_size_ = 0;

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
            std::ostringstream types;
            
            for(size_t i=0;i<xs.size();i++) {
                std::string type = data_type_to_opencl_type(xs[i].dtype());
                std::string suffix = "(" + type + "," + std::to_string(i) + ") ";
                types << "typedef " << type << " typeof_x" << i <<";\\\n";
                PARAMS << "PARAM_INPUT" << suffix;
                PREPARE_LOAD_INPUT_ALL << "PREPARE_LOAD_INPUT" << suffix << ";\\\n";
                LOAD_INPUT_ALL << "LOAD_INPUT(" << i << ");\\\n";
            }

            for(size_t i=0;i<ys.size();i++) {
                std::string type = data_type_to_opencl_type(ys[i].dtype());
                std::string ptype = data_type_to_opencl_param_type(ys[i].dtype());
                std::string suffix_out = "(" + type + "," + ptype + "," + std::to_string(i) + ") ";
                std::string suffix = "(" + type + "," + std::to_string(i) + ") ";
                types << "typedef " << type << " typeof_y" << i <<";\\\n";
                PARAMS << "PARAM_OUTPUT" << suffix_out;
                REDUCE_INIT_ALL << "REDUCE_INIT"<<suffix << ";\\\n";
                LOAD_REDUCE_ALL << "LOAD_REDUCE("<<i<<");\\\n";
                SAVE_REDUCE_ALL << "SAVE_REDUCE("<<i<<");\\\n";
                LOAD_REDUCED_SAVE_GLOBAL_ALL << "LOAD_REDUCED_SAVE_GLOBAL("<<i<<");\\\n";
            }


            REDUCE_INIT_ALL << format_code(reduce_init) << "\n";

            for(size_t i=0;i<params_count_;i++) {
                std::string type = data_type_to_opencl_param_type(target_type_);
                PARAMS<<", "<<type << " w" <<i;
                types << "typedef " << type << " typeof_w" << i <<";\\\n";
            }

            PREPARE_LOAD_INPUT_ALL << types.str();

            size_t total_reduce = 1;
            second_stage_stride_ = 1;
            for(unsigned i=0;i<reduce_dims.size();i++)
                total_reduce *= ref[i];

            int wg_size;
            bool small_reduction = 0;
            if(total_reduce >= 256) {
                wg_size = 256;
            }
            else if(total_reduce >= 128) {
                wg_size = 128;
            }
            else if(total_reduce >= 64) {
                wg_size = 64;
            }
            else {
                wg_size = 0;
                small_reduction = 1;
            }
            int items_per_wi,nd_range;
            if(small_reduction == 0) {
                items_per_wi = (total_reduce + wg_size - 1) / wg_size;
                if(items_per_wi >= 256) {
                    second_stage_stride_ = 256;
                }
                else if(items_per_wi >= 128) {
                    second_stage_stride_ = 128;
                }
                else if(items_per_wi >= 64) {
                    second_stage_stride_ = 64;
                }
                if(second_stage_stride_ > 1) {
                    nd_range = wg_size * second_stage_stride_;
                    items_per_wi = (total_reduce + nd_range - 1) / nd_range;
                    std::vector<TensorSpecs> big_ys;
                    std::vector<TensorSpecs> small_ys;
                    std::ostringstream code;
                    Shape full_shape(ys[0].shape().total_size(),second_stage_stride_);
                    Shape red_shape(ys[0].shape().total_size(),1);
                    for(unsigned i=0;i<ys.size();i++) {
                        big_ys.push_back(TensorSpecs(full_shape,ys[i].dtype()));
                        small_ys.push_back(TensorSpecs(red_shape,ys[i].dtype()));
                        code << "y" << i <<"=x"<<i<<";\n";
                        size_t size = (big_ys.back().memory_size() + 15) / 16*16;
                        ws_offsets_.push_back(std::make_pair(ws_size_,size));
                        ws_size_ += size;
                    }
                    second_stage_.reset(new PointwiseOperationBroadcastReduceImpl(
                        ctx,big_ys,small_ys,0,float_data,
                        code.str(),reduce_init,reduce));
                        
                }
                else {
                    int mpl = wg_size * items_per_wi;
                    nd_range = (total_reduce + mpl - 1) / mpl * wg_size;
                }
            }
            else {
                items_per_wi = total_reduce;
                nd_range = 1; 
            }
      
//#define DEBUG_2STAGE            
#ifdef DEBUG_2STAGE
            std::cerr << "Items per thread/wg_size/nd_range:" << items_per_wi << "/" << wg_size << "/" << nd_range<< std::endl;
#endif            
           
            cl::Program const &prog = gpu::Cache::instance().get_program(ctx,  "pointwise_broadcast_reduce",
                                                                               "REDUCE_DIMS",reduce_dims.size(),
                                                                               "SMALL_REDUCTION",small_reduction,
                                                                               "DIMS",ref.size(),
                                                                               "WG_SIZE",wg_size,
                                                                               "ITEMS_PER_WI",items_per_wi,
                                                                               "TWO_STAGE_REDUCTION",(second_stage_stride_ == 1 ? 0 : 1),
                                                                               "#PARAMS",PARAMS.str(),
                                                                               "#PREPARE_LOAD_INPUT_ALL",PREPARE_LOAD_INPUT_ALL.str(),
                                                                               "#REDUCE_INIT_ALL",REDUCE_INIT_ALL.str(),
                                                                               "#LOAD_INPUT_ALL",LOAD_INPUT_ALL.str(),
                                                                               "#LOAD_REDUCE_ALL",LOAD_REDUCE_ALL.str(),
                                                                               "#SAVE_REDUCE_ALL",SAVE_REDUCE_ALL.str(),
                                                                               "#LOAD_REDUCED_SAVE_GLOBAL_ALL",LOAD_REDUCED_SAVE_GLOBAL_ALL.str(),
                                                                               "#REDUCE",format_code(reduce),
                                                                               "#CALC",format_code(compute_code));
            kernel_ = cl::Kernel(prog,"exec");

            cl::NDRange range;
            cl::NDRange wg_range;
            int zero = reduce_dims.size();
            if(zero == 0) {
                range = get_broadcast_ndrange(ref);
                wg_range = cl::NullRange;
            }
            else {
                range = get_broadcast_reduce_ndrange(ref,zero,non_reduce_dims.size(),nd_range);
                wg_range = wg_size == 0 ? cl::NullRange : cl::NDRange(wg_size,1,1);
            }
            range_ = range;
            wg_range_ = wg_range;
            wg_size_ = wg_size;
            ref_ = ref;
            xs_specs_ = xs;
            ys_specs_ = ys;
            strides_ = std::move(strides);
        }
    private:
        size_t ws_size_;
        std::vector<TensorSpecs> xs_specs_,ys_specs_;
        std::vector<Shape> strides_;
        std::vector<std::pair<size_t,size_t> > ws_offsets_;
        size_t params_count_;
        size_t second_stage_stride_;
        DataType target_type_;
        size_t wg_size_;
        cl::NDRange range_,wg_range_;
        cl::Kernel kernel_;
        Shape ref_;
        std::unique_ptr<PointwiseOperationBroadcastReduceImpl> second_stage_;
    };


    ///
    /// Create objects:
    ///
    /// \param xs - vector of input tensor specs - such tensors are expected to be given to enqueue
    /// \param ys - vector of output tenorr specs - such tensors are expectred to be give to enqueue
    //  \param weights_count - size of parameters vector in enqueue
    /// \param weights_type - type of weights parameters as provided
    ///
    /// \param compute_code - OpenCL code to compute values. You can use x0, x1, ... xN as input values for each x for xs
    /// y0,.., yN for each output and w0,...,wN for each weight. For example "y0 = x0 + w0 * x1;"
    ///
    /// \param reduce_init - initalization of reduction variables `reduce_yN` for example "reduce_y0 = 0;" or "reduce_y0=-FLT_MAX;"
    /// \param reduce - code for sum reduction "reduce_y0 += y0" or max reduction "reduce_y0 = max(reduce_y0,y0)"
    ///
    std::unique_ptr<PointwiseOperationBroadcastReduce> PointwiseOperationBroadcastReduce::create(
                    Context &ctx,
                    std::vector<TensorSpecs> xs,
                    std::vector<TensorSpecs> ys,
                    int weights_count,DataType weights_type,
                    std::string const &compute_code,
                    std::string const &reduce_init,
                    std::string const &reduce)
    {
        std::unique_ptr<PointwiseOperationBroadcastReduce> r(new PointwiseOperationBroadcastReduceImpl(
                    ctx,xs,ys,
                    weights_count,weights_type,
                    compute_code,reduce_init,reduce));
        return r;
                                                                                                                
    }

    void pointwise_operation_broadcast_reduce(  std::vector<Tensor> xs,
                                                std::vector<Tensor> ys,
                                                std::vector<double>  ws,
                                                std::string const &compute,
                                                std::string const &reduce_init,
                                                std::string const &reduce,
                                                ExecutionContext const &e)
    {
        Context ctx(e);
        std::vector<TensorSpecs> xspec,yspec;
        std::vector<double> alpha,beta;
        for(auto const &x:xs) {
            xspec.push_back(x.specs());
        }
        for(auto const &y:ys) {
            yspec.push_back(y.specs());
            alpha.push_back(1.0);
            beta.push_back(0.0);
        }
        auto op = PointwiseOperationBroadcastReduce::create(
                            ctx,xspec,yspec,
                            ws.size(),ys[0].dtype(),compute,reduce_init,reduce);
        Tensor workspace;
        if(op->workspace() > 0)
            workspace = Tensor(ctx,Shape(op->workspace()),uint8_data);
        op->enqueue(xs,ys,workspace,ws,alpha,beta,e);
    }


} // core
} // dlprim

