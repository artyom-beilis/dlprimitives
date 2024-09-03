///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
void atomic_addf(__global volatile float *ptr,float v)
{
#if defined(__opencl_c_ext_fp32_global_atomic_add)
    atomic_fetch_add((__global volatile atomic_float *)ptr,v);
#elif defined(cl_intel_subgroups)
    __global atomic_int *p = (__global atomic_int *)(ptr);
    int prev,newv;
    do {
        prev = atomic_load(p);
        newv = as_int(as_float(prev) + v);
    } while(! atomic_compare_exchange_weak(p,&prev,newv));
#elif defined(__NV_CL_C_VERSION)
    float prev;
    asm volatile(
        "atom.global.add.f32 %0, [%1], %2;"
        : "=f"(prev)
        : "l"(ptr) , "f"(v)
        : "memory"
    );
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

