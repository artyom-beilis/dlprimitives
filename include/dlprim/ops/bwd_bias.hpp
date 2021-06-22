#pragma once
#include <dlprim/operator.hpp>
namespace dlprim {
    class BWBias {
    public:
        BWBias(Context &ctx,int batch,int rows_columns=1,DataType dt=float_data);
        ~BWBias();
        void backward(Tensor &dy,Tensor &dw,Tensor &ws,float beta,ExecutionContext const &e)
        {
            DLPRIM_CHECK(dy.shape().size() >= 2);
            DLPRIM_CHECK(dy.shape()[0] <= size_t(batch_));
            DLPRIM_CHECK(dy.shape().size_no_batch() == size_t(rows_columns_) * dy.shape()[1]);
            DLPRIM_CHECK(dw.shape().size() == 1);
            DLPRIM_CHECK(dw.shape().total_size() == dy.shape()[1]);
            if(ctx_.is_cpu_context())
                backward_cpu(dy,dw,beta);
            else
                backward_gpu(dy,dw,ws,beta,e);
        }
        size_t workspace(int features) const;
        int batch() const
        {
            return batch_;
        }
        int rows_columns() const
        {
            return rows_columns_;
        }
    private:
        void backward_cpu(Tensor &dy,Tensor &dw,float beta);
        void backward_gpu(Tensor &dy,Tensor &dw,Tensor &ws,float beta,ExecutionContext const &e);
        Context ctx_;

        int batch_;
        int rows_columns_;
        int wg_;
        int items_per_wi_;

        int wg2_;
        int items_per_wi2_;
        int size2_;

        bool two_stage_reduction_;

        cl::Kernel kernel_;
        cl::Kernel kernel2_;
    };
}
