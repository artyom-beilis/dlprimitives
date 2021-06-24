__kernel
void adam(int size,float beta1,float beta2,float inv_b1,float inv_b2,float lr,float weight_decay,float eps,
          __global float *p,int p_offset,
          __global float *g,int g_offset,
          __global float *m,int m_offset,
          __global float *v,int v_offset)
{
    int i = get_global_id(0);
    if(i >= size)
        return;
    p += p_offset;
    g += g_offset;
    m += m_offset;
    v += v_offset;

    float grad = g[i] + weight_decay * p[i];
    float m_next = beta1 * m[i] + (1-beta1) * grad;
    float v_next = beta2 * v[i] + (1-beta2) * grad * grad;
    float m_top = m_next * inv_b1;
    float v_top = v_next * inv_b2;
    float p_next = p[i] - lr * m_top / (sqrt(v_top) + eps);

    m[i] = m_next;
    v[i] = v_next;
    p[i] = p_next;
}
