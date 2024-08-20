///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
__kernel
void update_sums(int N,
                 __global float const * restrict cur_mean,ulong  cur_mean_offset,
                 __global float const * restrict cur_var ,ulong  cur_var_offset,
                 __global float * restrict run_mean,ulong  run_mean_offset,
                 __global float * restrict run_var ,ulong  run_var_offset,
                 float cur_mean_factor,float run_mean_factor,
                 float cur_var_factor, float run_var_factor)
{
    int p = get_global_id(0);
    if(p >= N)
        return;
    run_mean += run_mean_offset;
    run_var  += run_var_offset;
    cur_mean += cur_mean_offset;
    cur_var  += cur_var_offset;

    run_mean[p] = cur_mean[p] * cur_mean_factor + run_mean[p] * run_mean_factor;
    run_var[p]  = cur_var[p]  * cur_var_factor  + run_var[p]  * run_var_factor;
}



__kernel
void var_gamma_to_a(int N,float eps,
              __global float const * var, ulong  var_offset,
              __global float const * gamma, ulong  gamma_offset,
              __global float *a,ulong  a_offset)
{
    int pos = get_global_id(0);
    if(pos >= N)
        return;
    var  += var_offset;
    gamma += gamma_offset;
    a += a_offset;
    float scale = 1.0f / sqrt(var[pos] + eps);
    if(gamma)
        scale *= gamma[pos];
    a[pos] = scale;
}




__kernel
void mean_var_to_a_b(int N,float eps,
                     __global float const * mean,ulong  mean_offset,
                     __global float const * var, ulong  var_offset,
                     __global float *a,ulong  a_offset,
                     __global float *b,ulong  b_offset)
{
    int pos = get_global_id(0);
    if(pos >= N)
        return;
    mean += mean_offset;
    var  += var_offset;
    a += a_offset;
    b += b_offset;
    float scale = 1.0f / sqrt(var[pos] + eps);
    float offset  = - mean[pos] * scale;
    a[pos] = scale;
    b[pos] = offset;
}

__kernel
void combine_mean_var_with_gamma_beta(
                     int N,float eps,
                     __global float const * mean,ulong  mean_offset,
                     __global float const * var, ulong  var_offset,
                     __global float const * gamma,ulong  gamma_offset,
                     __global float const * beta,ulong  beta_offset,
                     __global float *a,ulong  a_offset,
                     __global float *b,ulong  b_offset)
{
    int pos = get_global_id(0);
    if(pos >= N)
        return;
    mean += mean_offset;
    var  += var_offset;
    gamma += gamma_offset;
    beta += beta_offset;
    a += a_offset;
    b += b_offset;
    float scale = 1.0f / sqrt(var[pos] + eps);
    float offset  = - mean[pos] * scale;
    float G = gamma[pos];
    scale *= G;
    offset = offset * G + beta[pos];
    a[pos] = scale;
    b[pos] = offset;
}

__kernel
void compute_backward_factors(int N,int M,float eps,
                              __global float const *mean,ulong  mean_offset,
                              __global float const *varrstd, ulong  varrstd_offset,
                              __global float const *dy_sum, ulong  dy_sum_offset,
                              __global float const *dyx_sum,ulong  dyx_sum_offset,
                              __global float const *gamma_in,ulong  gamma_in_offset,
                              __global float *x_factor,ulong  x_factor_offset,
                              __global float *dy_factor,ulong  dy_factor_offset,
                              __global float *offset,ulong  offset_offset)
{
    int i = get_global_id(0);
    if(i >= N)
        return;
    x_factor += x_factor_offset;
    dy_factor += dy_factor_offset;
    offset += offset_offset; 
    mean += mean_offset;
    varrstd  += varrstd_offset;
    if(gamma_in)
        gamma_in += gamma_in_offset;
    dyx_sum += dyx_sum_offset;
    dy_sum  += dy_sum_offset;

    float one_by_M = 1.0f / M;
    float rsqrtsig;
    if(eps < 0)
        rsqrtsig = varrstd[i];
    else
        rsqrtsig = 1.0f / sqrt(varrstd[i] + eps);

    float gamma=1.0f;
    if(gamma_in)
        gamma = gamma_in[i];
    float mu = mean[i];
    float dys = dy_sum[i];
    float dsig = -0.5 * gamma * (dyx_sum[i] - mu * dys) * (rsqrtsig * rsqrtsig * rsqrtsig);
    float gamma_div_sigsqrt = gamma * rsqrtsig;
    float dmu = -dys * gamma_div_sigsqrt;
    float F_dy = gamma_div_sigsqrt;
    float F_x  = 2*dsig * one_by_M;
    float B = one_by_M * (dmu - dsig * 2 * mu);

    dy_factor[i] = F_dy;
    x_factor[i] = F_x;
    offset[i] = B;
}

#define DIM_B  2
#define DIM_F  1
#define DIM_RC 0

__kernel
void forward(int batches,int channels,int HW,
             __global float const *x,ulong  x_offset,
             __global float *y,      ulong  y_offset,
             __global float const *A,ulong A_offset,
             __global float const *B,ulong B_offset)
{
    int b  = get_global_id(DIM_B);
    int f  = get_global_id(DIM_F);
    int rc = get_global_id(DIM_RC);
    if(b >= batches || f >= channels || rc >= HW)
        return;
    int pos = (b * channels + f) * HW + rc;
    y[y_offset + pos] = x[x_offset + pos] * A[A_offset + f] + B[B_offset + f];
}

__kernel
void backward_test(int batches,int channels,int HW,
             __global float *dx,ulong  dx_offset,
             __global float const *dy,ulong  dy_offset,
             __global float const *a,ulong  a_offset,
             float factor)
{
    int b  = get_global_id(DIM_B);
    int f  = get_global_id(DIM_F);
    int rc = get_global_id(DIM_RC);
    if(b >= batches || f >= channels || rc >= HW)
        return;
    dx+=dx_offset;
    dy+=dy_offset;
    a+=a_offset;
    int pos = (b * channels + f) * HW + rc;
    float val = dy[pos] * a[f];
    if(factor == 0)
        dx[pos] = val;
    else
        dx[pos] = dx[pos]*factor + val;
}


__kernel
void backward_data(int batches,int channels,int HW,
             __global float const *x,  ulong  x_offset,
             __global float const *dy, ulong  dy_offset,
             __global float const *fx, ulong  fx_offset,
             __global float const *fdy,ulong  fdy_offset,
             __global float const *b,  ulong  b_offset,
             __global float *dx,       ulong  dx_offset,
             float factor)
{
    int batch  = get_global_id(DIM_B);
    int f  = get_global_id(DIM_F);
    int rc = get_global_id(DIM_RC);
    if(batch >= batches || f >= channels || rc >= HW)
        return;
    int pos = (batch * channels + f) * HW + rc;
    float grad =  fx[fx_offset + f] * x[x_offset + pos]  + fdy[fdy_offset + f] * dy[dy_offset + pos] + b[b_offset + f];
    if(factor == 0)
        dx[dx_offset + pos] = grad;
    else
        dx[dx_offset + pos] = dx[dx_offset + pos] * factor + grad;
}

__kernel
void backward_filter(int N,
                    __global float const *mean,ulong  mean_offset,
                    __global float const *var, ulong  var_offset,
                    __global float const *dy_sum, ulong  dy_sum_offset,
                    __global float const *dyx_sum,ulong  dyx_sum_offset,
                    __global float *dgamma,ulong  dgamma_offset,
                    __global float *dbeta,ulong  dbeta_offset,
                    float eps,
                    float factor_gamma,
                    float factor_beta)
{
    int i=get_global_id(0);
    if(i >= N)
        return;
    mean += mean_offset;
    var  += var_offset;
    dy_sum += dy_sum_offset;
    dyx_sum += dyx_sum_offset;

    float dys = dy_sum[i];

    if(dgamma) {
        dgamma += dgamma_offset;
        float dG = (dyx_sum[i] - mean[i]*dys) / sqrt(var[i] + eps); 
        if(factor_gamma == 0)
            dgamma[i] = dG;
        else
            dgamma[i] = dgamma[i]*factor_gamma + dG;
    }
    
    if(dbeta) {
        dbeta += dbeta_offset;
        if(factor_beta == 0)
            dbeta[i] = dys;
        else
            dbeta[i] = dbeta[i] * factor_beta + dys;
    }
}
