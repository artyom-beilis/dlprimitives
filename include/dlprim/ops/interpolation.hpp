///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/operator.hpp>
#include <tuple>
namespace dlprim {	
    namespace json { class value; }
    struct InterpolationConfig {
        double scale_y,scale_x;
        int out_h,out_w;
        bool align_corners;
        InterpolateType method;
        static InterpolationConfig from_json(json::value const &v);
    };


   
    class Interpolation : public Operator {
    public:
        
        Interpolation(Context &ctx,InterpolationConfig config = InterpolationConfig());
        virtual ~Interpolation();
        
        virtual char const *operator_type() const
        {
            return "Interpolation";
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
        template<typename OpHandle>
        void bilinear_fwd_bwd_cpu(Tensor &in,Tensor &out);
        float calc_bin_src(int p,float scale);
        float calc_bin_scale(float scale,int x_size,int y_size);
        std::tuple<int,int,float,float> calc_bin_src_weight(int dst_intex,float scale,int size);
        int calc_size(int input,double scale);
        Shape calc_size(Shape input);
        void forward_cpu(Tensor &a,Tensor &output);
        void backward_cpu(Tensor &dx,Tensor &dy);
        InterpolationConfig config_;
    };
} // namespace
 
