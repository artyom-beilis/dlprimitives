///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
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

