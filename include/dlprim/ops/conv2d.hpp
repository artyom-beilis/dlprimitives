#pragma once
#include <dlprim/operator.hpp>

namespace dlprim {	
    
    namespace gpu { class GEMM; }

	struct Convolution2DConfig {
		int channels_in = -1;
		int channels_out = -1;
		int kernel[2] = {1,1};
		int stride[2] = {1,1};
		int dilate[2] = {1,1};
		int pad[2] = {0,0};
		int groups = {1};
		bool bias = true;
		StandardActivations activation=StandardActivations::identity;
        static Convolution2DConfig from_json(json::value const &v);
	};

	class Convolution2D : public Operator {
	public:

        Convolution2D(Context &ctx,Convolution2DConfig const &cfg);
        virtual ~Convolution2D();

        virtual char const *operator_type() const
        {
            return "Convolution2D";
        }
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


        Shape get_output_shape(Shape const &in);
	protected:
        void get_gemm(Shape const &in,Shape const &out);
        int get_im2col_width();
        void forward_gpu(Tensor &in,Tensor &out,Tensor &M,Tensor *bias,ExecutionContext const &ctx);
        void forward_cpu(Tensor &in,Tensor &out,Tensor &M,Tensor *bias,void *ws);
        void im2col(Shape const &in,Shape const &outs,float *img_in,float *mat_in);

        void setup_depthwise_separable_conv();
        bool is_depthwise_separable_conv();
        static int get_opt_val(int v);
		
        Convolution2DConfig config_;
        DataType dtype_;
        std::unique_ptr<gpu::GEMM> gemm_;
        size_t ws_size_;
        size_t out_h_,out_w_;
        size_t in_h_,in_w_;

        constexpr static int ds_patch_rows = 2;
        constexpr static int ds_patch_cols = 2;
        bool use_ds_conv_;
        cl::Kernel conv_;
	};
} // namespace

