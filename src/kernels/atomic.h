void atomic_addf(__global volatile float *ptr,float v)
{
#if defined(cl_intel_subgroups)
    __global atomic_int *p = (__global atomic_int *)(ptr);
    int prev,newv;
    do {
        prev = atomic_load(p);
        newv = as_int(as_float(prev) + v);
    } while(! atomic_compare_exchange_weak(p,&prev,newv));
#else
    float oldv = *ptr;
    for(;;) {
        float newv = oldv + v;
        int prev = atomic_cmpxchg((__global volatile int *)(ptr),as_int(oldv),as_int(newv));
        if(prev == as_int(oldv))
            return;
        oldv = as_float(prev);
    }
#endif    
}

