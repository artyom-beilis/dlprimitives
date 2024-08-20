///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/operator.hpp>
namespace dlprim {	
    namespace json { class value; }

    class ReshapeBase : public Operator {
    public:
        ReshapeBase(Context const &);
        virtual ~ReshapeBase();
        virtual bool alias_generator()
        {
            return true;
        }

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out,
                             size_t &ws);


		virtual void forward(std::vector<Tensor> &,
                             std::vector<Tensor> &,
                             std::vector<Tensor> &,
                             Tensor &,
                             ExecutionContext const &)
        {
            return;
        }

        virtual void backward(std::vector<TensorAndGradient> &,
                              std::vector<TensorAndGradient> &t,
                              std::vector<TensorAndGradient> &,
                              Tensor &,
                              ExecutionContext const &)
        {
            return;
        }
        
        virtual Shape new_shape(Shape const &in) = 0;

    };
   
    struct FlattenConfig {
        static FlattenConfig from_json(json::value const &) { return FlattenConfig(); }
    };

    class Flatten : public ReshapeBase {
    public:
        
        Flatten(Context &ctx,FlattenConfig const &/*config*/ = FlattenConfig()) : ReshapeBase(ctx) {}
        virtual ~Flatten() {}
        
        virtual char const *operator_type() const
        {
            return "Flatten";
        }
        virtual Shape new_shape(Shape const &in)
        {
            Shape r(in[0],in.size_no_batch());
            return r;
        }
    };
    struct SqueezeConfig {
        std::vector<int> dims;
        bool all=false;
        static SqueezeConfig from_json(json::value const &);
    };

    class Squeeze : public ReshapeBase {
    public:
        
        Squeeze(Context &ctx,SqueezeConfig const &config = SqueezeConfig()) : ReshapeBase(ctx), cfg_(config) {}
        virtual ~Squeeze() {}
        
        virtual char const *operator_type() const
        {
            return "Squeeze";
        }
        virtual Shape new_shape(Shape const &in)
        {
            if(cfg_.all)
                return in.squeeze();
            else
                return in.squeeze(cfg_.dims);
        }
        SqueezeConfig cfg_;
    };

    struct ReshapeConfig {
        std::vector<int> dims;
        static ReshapeConfig from_json(json::value const &);
    };

    class Reshape : public ReshapeBase {
    public:
        
        Reshape(Context &ctx,ReshapeConfig const &config = ReshapeConfig()) : ReshapeBase(ctx), cfg_(config) {}
        virtual ~Reshape() {}
        
        virtual char const *operator_type() const
        {
            return "Reshape";
        }
        virtual Shape new_shape(Shape const &in)
        {
            return in.reshape(cfg_.dims);
        }
        ReshapeConfig cfg_;
    };

} // namespace
 
