#pragma once
#include <dlprim/operator.hpp>

namespace dlprim {	
    namespace json { class value; }
    struct SoftMaxConfig {
        static SoftMaxConfig from_json(json::value const &) { return SoftMaxConfig(); }
    };
    
    class SoftMax : public Operator {
    public:
        SoftMax(Context &ctx,SoftMaxConfig const &cfg=SoftMaxConfig());
        virtual ~SoftMax();

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out);

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &workspace,
                             ExecutionContext const &ctx);

    private:
   		void forward_gpu(Tensor &input,Tensor &output,ExecutionContext const &ctx);
        void forward_cpu(Tensor &input,Tensor &output);
        void setup_kernel(int sm_range);
        DataType dtype_;
        cl::Kernel kernel_;
        int wg_size_;
        int items_per_wi_;
        int sm_range_;
        int nd_range_;
    };

    StandardActivations activation_from_json(json::value const &v);

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

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out);

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &workspace,
                             ExecutionContext const &ctx);


        
    private:
   		void forward_gpu(Tensor &a,Tensor &b,Tensor &output,ExecutionContext const &ctx);
        void forward_cpu(Tensor &a,Tensor &b,Tensor &output);
        ElementwiseConfig config_;
        DataType dtype_;
        cl::Kernel kernel_;
    };

    struct PoolingBase {
        enum Mode {
            max = 0,
            avg = 1
        };

        Mode mode = max;

        static PoolingBase from_json(json::value const &v);
    };

    struct Pooling2DConfig : public PoolingBase {
        int kernel[2] = {1,1};
        int pad[2] = {0,0};
        int stride[2]={1,1};
        bool count_include_pad = false;
        static Pooling2DConfig from_json(json::value const &v);
    };

    class Pooling2D : public Operator {
    public:
        Pooling2D(Context &ctx,Pooling2DConfig config = Pooling2DConfig());
        virtual ~Pooling2D();

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out);

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &workspace,
                             ExecutionContext const &ctx);

    private:
        Shape calc_shape(Shape ins);
        int calc_output_size(int in_size,int dim);
   		void forward_gpu(Tensor &in,Tensor &output,ExecutionContext const &ctx);
        
        template<typename Dtype,typename ReduceOpts>
        void forward_cpu(Tensor &in,Tensor &output,ReduceOpts rop);
        
        template<typename T>
        struct MaxRedcue;
        template<typename T>
        struct AveReduceFull;
        template<typename T>
        struct AveReduceValid;
        
        
        struct ReduceMax;
        struct ReduceAve;
        struct NormIdent;
        struct NormPartIdent;
        struct NormAve;
        struct NormPartAve;
        
        Pooling2DConfig config_;
        DataType dtype_;
        cl::Kernel kernel_;
        int wg_size_;
    };
    
    
}
