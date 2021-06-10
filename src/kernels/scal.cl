__kernel
void sscal(int size,float scale,__global float *p,int p_off)
{
    int pos = get_global_id(0);
    if(pos >= size)
        return;
    p+=p_off;
    if(scale == 0)
        p[pos] = 0;
    else
        p[pos] = p[pos] * scale;
}
