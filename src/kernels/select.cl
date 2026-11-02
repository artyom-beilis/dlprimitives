#include "broadcast_dims.h"

#if META >= 6
#include "atomic.h"
#define atomic_add atomic_addf
#endif

#define _POS(x) ulong x##pos=get_offset(pos,x##stride,x##off)

inline ulong get_offset(Shape s, Shape strides, ulong offset)
{
  ulong r = offset;
#pragma unroll
  for (int i = 0; i < DIMS; i++) {
    r += s.s[i] * strides.s[i];
  }
  return r;
}

inline bool valid_pos(Shape pos, Shape limits)
{
#pragma unroll
  for (int i = 0; i < DIMS; i++)
    if (pos.s[i] >= limits.s[i])
      return 0;
  return 1;
}

__kernel void sel(
  Shape limit,
  __global const dtype *vx, ulong xoff,
  __global const itype *__restrict idx, ulong ioff,
  __global dtype *vy, ulong yoff,
  Shape xstride, Shape istride, Shape ystride,
  long bound, long lbound, long dim,
#if (META&1)==1
  dtype alpha,
#define SCL *alpha
#else
#define SCL
#endif
  __global int *flag, ulong foff
)
{
  Shape pos = get_pos_broadcast(limit);
  if (!valid_pos(pos, limit)) {
    return;
  }
  long i = (long)idx[get_offset(pos, istride, ioff)];
  if (i < 0) i += lbound;
  if (i >= bound || i < 0) {
    if(foff!=ULONG_MAX)flag[foff] = -1;
    return;
  }
  RPOS;
  pos.s[dim] = (ulong)i;
  SPOS;
  SELOP(vy[ypos], vx[xpos] SCL);
}