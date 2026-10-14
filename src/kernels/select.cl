#include "broadcast_dims.h"

#if META >= 12
#include "atomic.h"
#define atomic_add atomic_addf
#endif

#if (META&4) == 0
#define SELOP(x,y) x=y
#else
#define SELOP(x,y) atomic_add(&x,y)
#endif

#define _POS(x) ulong x##pos=get_offset(pos,x##stride,x##off)
#if (META&2) == 0
#define RPOS _POS(y)
#define SPOS _POS(x)
#else
#define RPOS _POS(x)
#define SPOS _POS(y)
#endif

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
  ulong bound, ulong dim,
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
  itype i = idx[get_offset(pos, istride, ioff)];
  if (i >= bound) {
    if(flag)flag[foff] = -1;
    return;
  }
  RPOS;
  pos.s[dim] = i;
  SPOS;
  SELOP(vy[ypos], vx[xpos] SCL);
}