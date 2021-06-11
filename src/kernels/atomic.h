void atomic_addf(__global volatile float *ptr,float v)
{
    float oldv = *ptr;
    for(;;) {
        float newv = oldv + v;
        float prev = as_float(atomic_cmpxchg((__global volatile int *)(ptr),as_int(oldv),as_int(newv)));
        if(prev == oldv)
            return;
        oldv = prev;
    }
}

