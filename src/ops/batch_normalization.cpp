#include <dlprim/ops/batch_normalization.hpp>
#include <dlprim/ops/conv2d.hpp>
#include <dlprim/json.hpp>
#include <cmath>
#include <my_cblas.hpp>
#include <iostream>

namespace dlprim {
        BatchNorm2D::BatchNorm2D(Context &ctx,BatchNorm2DConfig const &config,DataType dt) :
            Operator(ctx),
            config_(config),
            dtype_(dt)
        {
        }

        BatchNorm2DConfig BatchNorm2DConfig::from_json(json::value const &v) 
        {
            BatchNorm2DConfig cfg;
            cfg.features = v.get<int>("features",cfg.features);
            cfg.eps = v.get<float>("eps",cfg.eps);
            cfg.momentum = v.get<float>("momentum",cfg.momentum);
            cfg.affine = v.get<bool>("affine",cfg.affine);
            cfg.use_global_stats = v.get<bool>("use_global_stats",cfg.use_global_stats);
            return cfg;
        }

        BatchNorm2D::~BatchNorm2D() {}
        void BatchNorm2D::setup(std::vector<TensorSpecs> const &in,
                                std::vector<TensorSpecs> &out,
                                std::vector<TensorSpecs> &parameters,
                                size_t &workspace)
        {
            DLPRIM_CHECK(in.size()==1);
            DLPRIM_CHECK(in[0].shape().size() == 4);
            if(config_.features == -1) {
                config_.features = in[0].shape()[1];
            }
            else {
                DLPRIM_CHECK(config_.features ==  int(in[0].shape()[1]));
            }
            out = in;
            parameters.clear();
            Shape std_shape(config_.features);
            parameters.push_back(TensorSpecs(std_shape,dtype_,false)); // mean
            parameters.push_back(TensorSpecs(std_shape,dtype_,false)); // variance
            if(config_.affine) {
                parameters.push_back(TensorSpecs(std_shape,dtype_));
                parameters.push_back(TensorSpecs(std_shape,dtype_));
            }
            Convolution2DConfig cfg;
            cfg.channels_in = cfg.channels_out = cfg.groups = config_.features;
            cfg.bias=true;
            conv_.reset(new Convolution2D(ctx_,cfg));
            conv_->mode(this->mode());
            std::vector<TensorSpecs> conv_out,conv_params;
            size_t conv_ws = 0;
            conv_->setup(in,conv_out,conv_params,conv_ws);
            workspace = conv_ws + config_.features * sizeof(float);
            if(mode() != CalculationsMode::predict) {
                current_mean_ = Tensor(ctx_,std_shape,dtype_);
                current_var_ = Tensor(ctx_,std_shape,dtype_);
            }
            combined_scale_ = Tensor(ctx_,Shape(config_.features,1,1,1),dtype_);
            combined_bias_ = Tensor(ctx_,std_shape,dtype_);
        }
        void BatchNorm2D::mode(CalculationsMode m)
        {
            Operator::mode(m);
            if(conv_)
                conv_->mode(m);
        }
        
        void BatchNorm2D::reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out)
        {
            conv_->reshape(in,out);
        }
		void BatchNorm2D::forward(std::vector<Tensor> &input,
                                  std::vector<Tensor> &output,
                                  std::vector<Tensor> &parameters,
                                  Tensor &workspace,
                                  ExecutionContext const &e)
        {
            DLPRIM_CHECK(parameters.size() == 2u * (1 + config_.affine));
            if(mode() == CalculationsMode::train && !config_.use_global_stats) {
                get_batch_stats(input[0],current_mean_,current_var_,e.first_context());
                update_sums(input[0].shape().total_size() / config_.features, current_mean_,current_var_,parameters[0],parameters[1],e.middle_context());
                if(config_.affine)
                    compute_conv_parameters(current_mean_,current_var_,&parameters.at(2),&parameters.at(3),e.middle_context());
                else
                    compute_conv_parameters(current_mean_,current_var_,nullptr,nullptr,e.middle_context());
            }
            else {
                if(config_.affine)
                    compute_conv_parameters(parameters.at(0),parameters.at(1),&parameters.at(2),&parameters.at(3),e.first_context());
                else
                    compute_conv_parameters(parameters.at(0),parameters.at(1),nullptr,nullptr,e.first_context());
            }
            std::vector<Tensor> tmp{ combined_scale_, combined_bias_ };
            conv_->forward(input,output,tmp,workspace,e.last_context());
        }

        void BatchNorm2D::compute_conv_parameters(Tensor &mean,Tensor &var,Tensor *at,Tensor *bt,ExecutionContext const &e)
        {
            if(ctx_.is_cpu_context()) {
                float *scale = combined_scale_.data<float>();
                float *bias  = combined_bias_.data<float>();
                float *m = mean.data<float>();
                float *v = var.data<float>();
                float *a = at ? at->data<float>() : nullptr;
                float *b = bt ? bt->data<float>() : nullptr;
                for(int i=0;i<config_.features;i++) {
                    float alpha = 1.0f / std::sqrt(v[i] + config_.eps);
                    float beta  = - m[i] * alpha;
                    if(a && b) {
                        alpha *= a[i];
                        beta = a[i]*beta + b[i];
                    }
                    scale[i] = alpha;
                    bias[i] = beta;
                }
            }
            else {
                DLPRIM_CHECK(!"Not implemented");
            }
        }

        void BatchNorm2D::update_sums(int M,Tensor &cm,Tensor &cv,Tensor &sm,Tensor &sv,ExecutionContext const &e)
        {
            if(ctx_.is_cpu_context()) {
                cblas_sscal(config_.features,(1.0f - config_.momentum),sm.data<float>(),1);
                cblas_saxpy(config_.features,config_.momentum,cm.data<float>(),1,sm.data<float>(),1);
                cblas_sscal(config_.features,(1.0f - config_.momentum),sv.data<float>(),1);
                float variance_factor = config_.momentum * M / (M - 1);
                cblas_saxpy(config_.features,variance_factor,cv.data<float>(),1,sv.data<float>(),1);
            }
            else {
                DLPRIM_CHECK(!"Not implemented");
            }
        }

        void BatchNorm2D::get_batch_stats(Tensor &x,Tensor &mean,Tensor &var,ExecutionContext const &/*e*/)
        {
            if(ctx_.is_cpu_context()) {
                float *m = mean.data<float>();
                float *v = var.data<float>();
                float *img = x.data<float>();
                size_t rc_size = x.shape()[2]*x.shape()[3];
                int M = x.shape()[0]*rc_size;
                float factor = 1.0f / M;
                for(int f=0;f<config_.features;f++) {
                    m[f] = 0;
                    v[f] = 0;
                    float s=0,s2=0;
                    for(unsigned b=0;b<x.shape()[0];b++) {
                        float *ptr = img + (b*config_.features + f)*rc_size;
                        for(unsigned rc=0;rc < rc_size;rc++) {
                            float val = *ptr++;
                            s+= val;
                            s2 += val*val;
                        }
                    }
                    s*=factor;
                    s2*=factor;
                    m[f] = s;
                    v[f] = s2 - s*s;
                }
            }
            else {
                DLPRIM_CHECK(!"Not implemented");
            }
        }

        void BatchNorm2D::cpu_backward_data(Tensor &x,Tensor &dx,Tensor &dy,float *mean,float *var,float *dy_sum,float *dyx_sum,float *gamma_in)
        {
            int M = x.shape().total_size() / config_.features;
            int RC =x.shape()[2]*x.shape()[3];
            float one_by_M = 1.0f / M;
            for(int i=0;i<config_.features;i++) {
                float sqrtsig = std::sqrt(var[i] + config_.eps);
                float gamma=1.0f;
                if(gamma_in)
                    gamma = gamma_in[i];
                float dsig = -0.5 * gamma * (dyx_sum[i] - mean[i] * dy_sum[i]) / (sqrtsig * sqrtsig * sqrtsig);
                float gamma_div_sigsqrt = gamma / sqrtsig;
                float dmu = -dy_sum[i] * gamma_div_sigsqrt;
                float F_dy = gamma_div_sigsqrt;
                float F_x  = 2*dsig * one_by_M;
                float B = one_by_M * (dmu - dsig * 2 * mean[i]);

                for(unsigned b=0;b<x.shape()[0];b++) {
                    float *x_p  =  x.data<float>() + b * RC * config_.features + i*RC;
                    float *dy_p = dy.data<float>() + b * RC * config_.features + i*RC;
                    float *dx_p = dx.data<float>() + b * RC * config_.features + i*RC;
                    for(int j=0;j<RC;j++) {
                        dx_p[j] += dy_p[j] * F_dy + x_p[j] * F_x + B;
                    }
                }
            }
        }

        void BatchNorm2D::backward( std::vector<TensorAndGradient> &input,
                                    std::vector<TensorAndGradient> &output,
                                    std::vector<TensorAndGradient> &parameters,
                                    Tensor &workspace,
                                    ExecutionContext const &ctx)
        {
            Tensor temp_diff(ctx_,Shape(config_.features));
            Tensor temp_bias(ctx_,Shape(config_.features));
            
            TensorAndGradient s,o;
            
            s.data = combined_scale_;
            s.diff = temp_diff;
            s.accumulate_gradient = 0.0;
            s.requires_gradient = true;

            o.data = combined_bias_;
            o.requires_gradient = true;
            o.accumulate_gradient = 0.0;
            o.diff = temp_bias;
                
            std::vector<TensorAndGradient> cp{s,o};
            auto tmp_inp = input;
            if(mode()==CalculationsMode::train)
                tmp_inp[0].requires_gradient = false;
            conv_->backward(tmp_inp,output,cp, workspace,ctx);

            if(config_.affine && parameters[3].requires_gradient) {
                if(ctx_.is_cpu_context()) {
                    if(parameters[3].accumulate_gradient == 0) {
                        memcpy(parameters[3].diff.data<float>(),temp_bias.data<float>(),sizeof(float)*config_.features);
                    }
                    else {
                       cblas_sscal(config_.features,parameters[3].accumulate_gradient,parameters[3].diff.data<float>(),1);
                       cblas_saxpy(config_.features,1.0f,temp_bias.data<float>(),1,parameters[3].diff.data<float>(),1);
                    }
                }
                else {
                    DLPRIM_CHECK(!"Not Implemented");
                }
            }

            if(ctx_.is_cpu_context()){
                float *dy_sum = temp_bias.data<float>();
                float *variance = nullptr;
                float *mean = nullptr;
                if(config_.use_global_stats || mode() == CalculationsMode::predict) {
                    variance = parameters[1].data.data<float>();
                    mean = parameters[0].data.data<float>();
                }
                else {
                    variance = current_var_.data<float>();
                    mean = current_mean_.data<float>();
                }
                float *dyx_sum = temp_diff.data<float>();
                if(config_.affine && parameters[2].requires_gradient) {
                    float *da = parameters[2].diff.data<float>();
                    float factor = parameters[2].accumulate_gradient;
                    for(int i=0;i<config_.features;i++) {
                        float val = (dyx_sum[i] - mean[i]*dy_sum[i]) / std::sqrt(variance[i] + config_.eps); 
                        if(factor == 0)
                            da[i] = val;
                        else
                            da[i] = da[i]*factor + val;
                    }
                }
                if(input[0].requires_gradient && mode()==CalculationsMode::train) {
                    float *dx = input[0].diff.data<float>();
                    if(input[0].accumulate_gradient == 0)
                        memset(dx,0,input[0].diff.memory_size());
                    else
                        cblas_sscal(input[0].diff.shape().total_size(),input[0].accumulate_gradient,dx,1);
                    float *gamma = nullptr;
                    if(config_.affine)
                        gamma = parameters[2].data.data<float>();
                    cpu_backward_data(input[0].data,input[0].diff,output[0].diff,
                        mean,variance,dy_sum,dyx_sum,gamma);
                }
            }
            else {
                DLPRIM_CHECK(!"Not implemented");
            }
        }


} // namesapce
