__kernel
void update_sums(int N,
                 __global float const restrict *cur_mean,int cur_mean_offset,
                 __global float const restrict *cur_var ,int cur_var_offset,
                 __global float const restrict *run_mean,int run_mean_offset,
                 __global float const restrict *run_var ,int run_var_offset
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
void mean_var_to_a_b(int N,float eps,
                     __global float const * mean,int mean_offset,
                     __global float const * var, int var_offset,
                     __global float *a,int a_offset,
                     __global float *b,int b_offset)
{
    int pos = get_global_id(0);
    if(pos >= N)
        return;
    mean += mean_offset;
    var  += var_offset;
    a += a_offset;
    b += b_offset;
    float scale = 1.0f / sqrt(var[pos] + eps);
    float offset  = - mean[pos] * alpha;
    a[pos] = scale;
    b[pos] = offset;
}

__kernel
void combine_mean_var_with_gamma_beta(
                     int N,float eps,
                     __global float const * mean,int mean_offset,
                     __global float const * var, int var_offset,
                     __global float const * gamma,int gamma_offset,
                     __global float const * beta,int beta_offset,
                     __global float *a,int a_offset,
                     __global float *b,int b_offset)
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
    float offset  = - mean[pos] * alpha;
    float G = gamma[pos];
    scale += G;
    offset = offset * G + beta[pos];
    a[pos] = scale;
    b[pos] = offset;
}

__kernel
void compute_backward_factors(int N,int M,
                              __global float const *mean,int mean_offset,
                              __global float const *var, int var_offset,
                              __global float const *dy_sum, int dy_sum_offset,
                              __global float const *dxy_sum,int dxy_sum_offset,
                              __global float const *gamma_in,int gamma_in_offset,
                              __global float *x_factor,int x_factor_offset,
                              __global float *dy_factor,int dy_factor_offset,
                              __global float *offset,int offset_offset)
{
    int i = get_global_id(0);
    if(i >= N)
        return;
    x_factor += x_factor_offset;
    dy_factor += dy_factor_offset;
    offset += offset_offset; 
    mean += mean_offset;
    var  += var_offset;
    if(gamma_in != NULL)
        gamma_in += gamma_in_offset;
    dxy_sum += dxy_sum_offset;
    dy_sum  += dy_sum_offset;

    float one_by_M = 1.0f / M;
    float sqrtsig = std::sqrt(var[i] + config_.eps);
    float gamma=1.0f;
    if(gamma_in)
        gamma = gamma_in[i];
    float mu = mean[i];
    float dys = dy_sum[i];
    float dsig = -0.5 * gamma * (dyx_sum[i] - mu * dys[i]) / (sqrtsig * sqrtsig * sqrtsig);
    float gamma_div_sigsqrt = gamma / sqrtsig;
    float dmu = -dys * gamma_div_sigsqrt;
    float F_dy = gamma_div_sigsqrt;
    float F_x  = 2*dsig * one_by_M;
    float B = one_by_M * (dmu - dsig * 2 * mu);

    dy_factor[i] = F_dy;
    x_factor[i] = F_x;
    offset[i] = B;
}

#define DIM_B  0
#define DIM_F  1
#define DIM_RC 2

__kernel
void forward(int batches,int channels,int HW,
             __global float const *x,int x_offset,
             __global float *y,      int y_offset,
             __global float const *a,int a_offset,
             __global float const *b,int b_offset
{
    int b  = get_global_id(DIM_B);
    int f  = get_global_id(DIM_F);
    int rc = get_global_id(DIM_RC);
    if(b >= batches || f >= channels || rc >= HW)
        return;
    int pos = (b * channels + f) * HW + rc;
    y[y_offset + pos] = x[x_offset + pos] * a[a_offset + f] + b[b_offset + f];
}

__kernel
void backward_data(int batches,int channels,int HW,
             __global float const *x,  int x_offset,
             __global float cosst *dy, int dy_offset,
             __global float const *fx, int fx_offset,
             __global float const *fdy,int fdy_offset
             __global float const *b,  int fb_offset,
             __global float *dx,       int dx_offset,
             float factor)
{
    int b  = get_global_id(DIM_B);
    int f  = get_global_id(DIM_F);
    int rc = get_global_id(DIM_RC);
    if(b >= batches || f >= channels || rc >= HW)
        return;
    int pos = (b * channels + f) * HW + rc;
    float grad =  fx[fx_offset + f] * x[x_offset + pos]  + fdy[fdy_offset + f] * dy[dy_offset + pos] + b[b_offset + pos];
    if(factor == 0)
        dx[dx_offset + pos] = grad;
    else
        dx[dx_offset + pos] = dx[dx_offset + pos] * factor + grad;
}

__kernel
void backward_filter(int N,
                    __global float const *mean,int mean_offset,
                    __global float const *var, int var_offset,
                    __global float const *dy_sum, int dy_sum_offset,
                    __global float const *dyx_sum,int dyx_sum_offset
                    __global float *dgamma,int dgamma_offset,
                    __global float *dbeta,int dbeta_offset,
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
        float dG = (dyx_sum[i] - mean[i]*dys) / std::sqrt(var[i] + eps); 
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
