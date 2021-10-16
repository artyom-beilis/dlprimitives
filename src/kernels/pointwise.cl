#include "defs.h"

__kernel void exec(ulong total  PARAMS)
{
    ulong index=get_global_id(0);
    if(index>=total)
        return;
    LOADS
    CALC
    SAVES
}

