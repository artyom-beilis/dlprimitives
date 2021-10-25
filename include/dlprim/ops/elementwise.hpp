#pragma once
#include <dlprim/operator.hpp>
namespace dlprim {	
    namespace json { class value; }
    struct ElementwiseConfig {
        enum Operation {
            elementwise_sum,
            elementwise_prod,
            elementwise_max
        };
        
        Operation op = elementwise_sum;
        float coeff[2] = {1.0f,1.0f};
        StandardActivations activation = StandardActivations::identity;

        static ElementwiseConfig from_json(json::value const &v);
    };
   
    class Elementwise : public Operator {
    public:
        
        Elementwise(Context &ctx,ElementwiseConfig config = ElementwiseConfig());
        virtual ~Elementwise();
        
        virtual char const *operator_type() const
        {
            return "Elementwise";
        }

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out,
                             size_t &ws);

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &workspace,
                             ExecutionContext const &ctx);
        
        virtual void backward(std::vector<TensorAndGradient> &input,
                              std::vector<TensorAndGradient> &output,
                              std::vector<TensorAndGradient> &parameters,
                              Tensor &workspace,
                              ExecutionContext const &ctx);


        
    private:
   		void forward_gpu(Tensor &a,Tensor &b,Tensor &output,ExecutionContext const &ctx);
        void forward_cpu(Tensor &a,Tensor &b,Tensor &output);

        template<int index>
        struct StridePos;

        template<int dim,typename F>
        void loop_strides_dim(Shape s,float *a,Shape a_strides,float *b,Shape b_strides,float *c,F const &func);

        template<typename F>
        void loop_strides(Shape s,float *a,Shape a_strides,float *b,Shape b_strides,float *c,F const &func);

        template<typename F,typename R>
        void loops_reduce(Shape s,float *a,Shape as,float *r,Shape rs,F const &func,R const &reduce);
        template<typename F,typename R>
        void loops_reduce(Shape s,float *a,Shape as,float *b,Shape bs,float *r,Shape rs,F const &func,R const &reduce);
        template<typename F,typename R>
        void loops_reduce(Shape s,float *a,Shape as,float *b,Shape bs,float *c,Shape cs,float *r,Shape rs,F const &func,R const &reduce);


        template<int dim,typename F,typename R>
        void loops_reduce_dim(Shape s,float *a,Shape as,float *r,Shape rs,F const &func,R const &reduce);
        template<int dim,typename F,typename R>
        void loops_reduce_dim(Shape s,float *a,Shape as,float *b,Shape bs,float *r,Shape rs,F const &func,R const &reduce);
        template<int dim,typename F,typename R>
        void loops_reduce_dim(Shape s,float *a,Shape as,float *b,Shape bs,float *c,Shape cs,float *r,Shape rs,F const &func,R const &reduce);



        
        void backward_cpu(Tensor &a,Tensor &da,
                          Tensor &b,Tensor &db,
                          Tensor &c,Tensor &dc,
                          bool l,bool r, 
                          float ba,float bb);

        void backward_gpu(Tensor &a,Tensor &da,
                          Tensor &b,Tensor &db,
                          Tensor &c,Tensor &dc,
                          bool l,bool r, 
                          float ba,float bb,
                          ExecutionContext const &e);

        ElementwiseConfig config_;
        DataType dtype_;
        cl::Kernel kernel_;
        cl::Kernel kernel_bwd_;
    };
} // namespace
 
