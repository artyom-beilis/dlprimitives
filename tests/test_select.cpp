#include <dlprim/core/common.hpp>
#include <dlprim/core/select.hpp>
#include "test.hpp"
#include <iostream>
#include <cstring>
#include <vector>
#include <cstdint>

#define TP(T) dp::TypeTraits<T>::data_type

namespace dp = dlprim;

// ---------------------------------------------------------------------------
// type names for printing
// ---------------------------------------------------------------------------
template<typename T> static const char* tname();
template<> const char* tname<float>()   { return "f32"; }
template<> const char* tname<int32_t>() { return "i32"; }
template<> const char* tname<int64_t>() { return "i64"; }
template<> const char* tname<int16_t>() { return "i16"; }
template<> const char* tname<int8_t>()  { return "i8";  }
template<> const char* tname<uint8_t>() { return "u8";  }

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static size_t flat(dp::Shape const& c, dp::Shape const& shape)
{
    size_t n = 0;
    for (int i = 0; i < c.size(); ++i)
        n = n * shape[i] + c[i];
    return n;
}

static dp::Shape unravel(size_t i, dp::Shape const& shape)
{
    auto c = shape;
    for (int k = (int)shape.size() - 1; k >= 0; --k) {
        c[k] = i % shape[k];
        i /= shape[k];
    }
    return c;
}

static dp::Tensor mk(dp::ExecutionContext const& q,
                     dp::Shape shape,
                     dp::DataType dtype,
                     void *ptr)
{
    dp::Context ctx(q);
    dp::Tensor t(ctx, shape, dtype);
    t.to_device(q, ptr);
    return t;
}

static dp::Tensor zeros(dp::ExecutionContext const& q, dp::Shape shape, dp::DataType dtype)
{
    dp::Context ctx(q);
    dp::Tensor t(ctx, shape, dtype);
    dp::core::fill_tensor(t, 0, q);
    return t;
}

static bool eq(dp::Tensor a, dp::Tensor b, dp::ExecutionContext const& q)
{
    a.to_host(q); b.to_host(q);
    return a.shape() == b.shape() && a.dtype() == b.dtype()
        && memcmp(a.host_data(), b.host_data(), a.memory_size()) == 0;
}

// default index vector of length K covering spread positions
template<typename I>
static std::vector<I> default_index(size_t bound, size_t K)
{
    std::vector<I> r(K);
    for (size_t i = 0; i < K; ++i) r[i] = (I)i;
    if (K > 0 && bound > 0) r[K-1] = (I)(bound - 1);
    return r;
}

// ---------------------------------------------------------------------------
// run helper (now exposes negIdx)
// ---------------------------------------------------------------------------
static void run(dp::ExecutionContext const& q,
                dp::Tensor& x, dp::Tensor& y, dp::Tensor& idx,
                uint64_t axis,
                dp::core::Select::Op op, dp::core::Select::Target tg,
                bool scal = false, double alpha = 0.0,
                dp::Tensor* flag = nullptr, bool negIdx = true)
{
    dp::Context ctx(q);
    auto s = dp::core::Select::create(
        ctx, (int)x.shape().size(), x.dtype(), idx.dtype(), op, tg, scal);
    s->enqueue(x, y, idx, axis, flag, alpha, q, negIdx);
}

template<typename D, typename I>
static void title(char const* name, dp::Shape const& shape, int axis)
{
    std::cout << name << ": dims=" << shape.size()
              << " shape=" << shape
              << " axis=" << axis
              << " D=" << tname<D>()
              << " I=" << tname<I>()
              << std::endl;
}

// ---------------------------------------------------------------------------
// index_select (ASSIGN / Target::X)
// ---------------------------------------------------------------------------
template<typename D, typename I>
void test_index_select(dp::ExecutionContext const& q,
                       dp::Shape xshape, int axis,
                       std::vector<I> index, bool negIdx = true)
{
    title<D, I>("index_select", xshape, axis);

    size_t K = index.size();
    auto yshape = xshape; yshape[axis] = K;

    auto xsize = xshape.total_size();
    auto ysize = yshape.total_size();

    std::vector<D> xd(xsize);
    for (size_t i = 0; i < xsize; ++i) xd[i] = (D)(i + 1);

    std::vector<D> yd(ysize);
    for (size_t out = 0; out < ysize; ++out) {
        auto c = unravel(out, yshape);
        auto ic = c;
        long iv = (long)index[c[axis]];
        if (negIdx && iv < 0) iv += (long)xshape[axis];
        ic[axis] = (size_t)iv;
        yd[out] = xd[flat(ic, xshape)];
    }

    auto x   = mk(q, xshape, TP(D), xd.data());
    auto idx = mk(q, {K}, TP(I), index.data());
    auto y   = zeros(q, yshape, TP(D));

    run(q, x, y, idx, axis, dp::core::Select::ASSIGN, dp::core::Select::X,
        /*scal=*/false, 0.0, nullptr, negIdx);

    auto ey = mk(q, yshape, TP(D), yd.data());
    TEST(eq(y, ey, q));
}

// ---------------------------------------------------------------------------
// gather: y.shape == idx.shape, non-axis dims may be smaller than x.shape
// ---------------------------------------------------------------------------
template<typename D, typename I>
void test_gather(dp::ExecutionContext const& q,
                 dp::Shape xshape, int axis, bool small = false)
{
    title<D, I>("gather", xshape, axis);

    auto yshape = xshape;
    yshape[axis] = std::min<size_t>(xshape[axis], 2);
    if (small)
        for (int i = 0; i < yshape.size(); ++i)
            if (i != axis && yshape[i] > 2)
                yshape[i] = 2;

    auto xsize = xshape.total_size();
    auto ysize = yshape.total_size();

    std::vector<D> xd(xsize);
    for (size_t i = 0; i < xsize; ++i) xd[i] = (D)(i + 1);

    std::vector<I> idxd(ysize);
    std::vector<D> yd(ysize);
    for (size_t out = 0; out < ysize; ++out) {
        auto c = unravel(out, yshape);
        size_t iv = (c[axis] + 1) % xshape[axis];
        idxd[out] = (I)iv;
        auto ic = c;
        ic[axis] = iv;
        yd[out] = xd[flat(ic, xshape)];
    }

    auto x   = mk(q, xshape, TP(D), xd.data());
    auto idx = mk(q, yshape, TP(I), idxd.data());
    auto y   = zeros(q, yshape, TP(D));

    run(q, x, y, idx, axis, dp::core::Select::ASSIGN, dp::core::Select::X);

    auto ey = mk(q, yshape, TP(D), yd.data());
    TEST(eq(y, ey, q));
}

// ---------------------------------------------------------------------------
// index_add (ADD / Target::Y)
// ---------------------------------------------------------------------------
template<typename D, typename I>
void test_index_add(dp::ExecutionContext const& q,
                    dp::Shape selfshape, int axis, double alpha = 1.0)
{
    title<D, I>("index_add", selfshape, axis);

    const size_t K = 4;
    auto sshape = selfshape; sshape[axis] = K;
    auto index = default_index<I>(selfshape[axis], K);

    auto ssize = sshape.total_size();
    auto ysize = selfshape.total_size();

    std::vector<D> srcd(ssize);
    for (size_t i = 0; i < ssize; ++i) srcd[i] = (D)(i + 1);

    std::vector<D> yd(ysize, D(0));
    for (size_t out = 0; out < ssize; ++out) {
        auto c = unravel(out, sshape);
        auto oc = c;
        oc[axis] = index[c[axis]];
        yd[flat(oc, selfshape)] += (D)(alpha * (double)srcd[out]);
    }

    auto src = mk(q, sshape, TP(D), srcd.data());
    auto idx = mk(q, {K}, TP(I), index.data());
    auto y   = zeros(q, selfshape, TP(D));

    run(q, src, y, idx, axis, dp::core::Select::ADD, dp::core::Select::Y,
        /*scal=*/true, alpha);

    auto ey = mk(q, selfshape, TP(D), yd.data());
    TEST(eq(y, ey, q));
}

// ---------------------------------------------------------------------------
// scatter_add (ADD / Target::Y, idx.shape == src.shape)
// ---------------------------------------------------------------------------
template<typename D, typename I>
void test_scatter_add(dp::ExecutionContext const& q,
                      dp::Shape selfshape, int axis, bool small = false)
{
    title<D, I>("scatter_add", selfshape, axis);

    auto sshape = selfshape;
    if (small)
        for (int i = 0; i < sshape.size(); ++i)
            if (i != axis && sshape[i] > 2)
                sshape[i] = 2;

    auto ssize = sshape.total_size();
    auto ysize = selfshape.total_size();

    std::vector<D> srcd(ssize);
    for (size_t i = 0; i < ssize; ++i) srcd[i] = (D)(i + 1);

    std::vector<I> idxd(ssize);
    std::vector<D> yd(ysize, D(0));
    for (size_t out = 0; out < ssize; ++out) {
        auto c = unravel(out, sshape);
        size_t iv = (c[axis] + 1) % selfshape[axis];
        idxd[out] = (I)iv;
        auto oc = c;
        oc[axis] = iv;
        yd[flat(oc, selfshape)] += srcd[out];
    }

    auto src = mk(q, sshape, TP(D), srcd.data());
    auto idx = mk(q, sshape, TP(I), idxd.data());
    auto y   = zeros(q, selfshape, TP(D));

    run(q, src, y, idx, axis, dp::core::Select::ADD, dp::core::Select::Y);

    auto ey = mk(q, selfshape, TP(D), yd.data());
    TEST(eq(y, ey, q));
}

// ---------------------------------------------------------------------------
// out-of-bound flag: valid / overflow / underflow, with negIdx option
// oob_dir: 0 = valid, +1 = overflow, -1 = underflow
// ---------------------------------------------------------------------------
template<typename D, typename I>
void test_oob(dp::ExecutionContext const& q,
    dp::Shape shape, int axis, int oob_dir, bool is_x = true,
    bool negIdx = true)
{
    char tit[32];
    sprintf(tit, "oob_%c(%s)", is_x ? 'X' : 'Y',
            oob_dir == 0 ? "valid" : (oob_dir > 0 ? "over" : "under"));
    title<D, I>(tit, shape, axis);

    auto small_shape = shape; small_shape[axis] = 2;

    auto& xshape = is_x ? shape : small_shape;
    auto& yshape = is_x ? small_shape : shape;

    auto xsize = xshape.total_size();
    auto isize = small_shape.total_size();

    std::vector<D> xd(xsize);
    for (size_t i = 0; i < xsize; ++i) xd[i] = (D)(i + 1);

    std::vector<I> idxd(isize, I(0));

    long bound = (long)shape[axis];
    long v;
    if (oob_dir == 0)      v = bound - 1;             // valid
    else if (oob_dir > 0)  v = bound;                 // overflow
    else if (negIdx)       v = -(bound + 1);          // underflow via negIdx
    else                   v = -1;                    // underflow raw negative

    idxd[0] = (I)0;
    idxd[1] = (I)v;

    auto x    = mk(q, xshape, TP(D), xd.data());
    auto idx  = mk(q, small_shape, TP(I), idxd.data());
    auto y    = zeros(q, yshape, TP(D));
    auto flag = zeros(q, {1}, TP(int32_t));

    run(q, x, y, idx, axis,
        dp::core::Select::ASSIGN,
        is_x ? dp::core::Select::X : dp::core::Select::Y,
        /*scal=*/false, 0.0, &flag, negIdx);

    flag.to_host(q);
    int32_t expected = (oob_dir != 0) ? -1 : 0;
    TEST(flag.data<int32_t>()[0] == expected);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc != 2) { std::cerr << "Usage: platform:device\n"; return 1; }
    try {
        dp::Context ctx(argv[1]);
        if (ctx.is_cpu_context()) return 0;
        auto q = ctx.make_execution_context();
        std::cout << ctx.name() << std::endl;

        // -------- index_select: dims 1..8, K=4 ----------------------
        test_index_select<float, int32_t>(q, {5},              0, default_index<int32_t>(5,4));
        test_index_select<float, int32_t>(q, {4,5},            0, default_index<int32_t>(4,4));
        test_index_select<float, int32_t>(q, {3,4,5},          1, default_index<int32_t>(4,4));
        test_index_select<float, int32_t>(q, {3,4,5,6},        2, default_index<int32_t>(5,4));
        test_index_select<float, int32_t>(q, {3,2,4,5,6},      2, default_index<int32_t>(4,4));
        test_index_select<float, int32_t>(q, {3,2,4,5,6,7},    3, default_index<int32_t>(5,4));
        test_index_select<float, int32_t>(q, {3,2,4,5,6,7,8},  4, default_index<int32_t>(6,4));
        test_index_select<float, int32_t>(q, {3,2,4,5,6,7,8,9},5, default_index<int32_t>(7,4));

        // -------- index_select: other data / index dtypes -----------
        test_index_select<float, int64_t>(q, {4,5,6}, 1, default_index<int64_t>(5,4));
        test_index_select<int32_t, int32_t>(q, {4,5,6}, 0, default_index<int32_t>(4,4));
        test_index_select<int32_t, int64_t>(q, {4,5,6}, 2, default_index<int64_t>(6,4));
        test_index_select<int64_t, int32_t>(q, {4,5,6}, 1, default_index<int32_t>(5,4));

        // -------- index_select: index integer widths ----------------
        // uint8 with bound >= 256
        test_index_select<float, uint8_t>(q, {256,4}, 0, {0,1,128,255});
        test_index_select<float, int8_t>(q, {100,4},  0, {0,1,98,99});
        test_index_select<float, int16_t>(q, {300,4}, 0, {0,1,255,299});

        // -------- gather --------------------------------------------
        test_gather<float, int32_t>(q, {3,4},     0);
        test_gather<float, int32_t>(q, {3,4,5},   1);
        test_gather<float, int32_t>(q, {3,4,5,6}, 3);
        test_gather<float, int64_t>(q, {3,4,5},   1);
        test_gather<int32_t, int32_t>(q, {3,4,5}, 2);
        // idx.shape[i] < x.shape[i] for i != axis
        test_gather<float, int32_t>(q, {3,4,5},   1, true);
        test_gather<float, int64_t>(q, {3,4,5,6}, 2, true);

        // -------- index_add -----------------------------------------
        test_index_add<float, int32_t>(q, {5,4},     0);
        test_index_add<float, int32_t>(q, {4,5},     1);
        test_index_add<float, int32_t>(q, {4,5,6},   0);
        test_index_add<float, int32_t>(q, {3,4,5},   1);
        test_index_add<float, int32_t>(q, {3,4,5},   2);
        test_index_add<float, int32_t>(q, {3,4,5,6}, 2);
        test_index_add<float, int64_t>(q, {3,4,5},   1);
        test_index_add<float, int32_t>(q, {4,5,6},   2, 0.25);

        // -------- scatter_add ---------------------------------------
        test_scatter_add<float, int32_t>(q, {3,4},     0);
        test_scatter_add<float, int32_t>(q, {3,4,5},   1);
        test_scatter_add<float, int32_t>(q, {3,4,5,6}, 3);
        test_scatter_add<float, int64_t>(q, {3,4,5},   1);
        test_scatter_add<int32_t, int32_t>(q, {3,4,5}, 1);
        // idx.shape[i] < y.shape[i] for i != axis
        test_scatter_add<float, int32_t>(q, {3,4,5},   1, true);
        test_scatter_add<float, int64_t>(q, {3,4,5,6}, 2, true);

        // -------- OOB flag: valid / overflow / underflow, with negIdx ----
        // Target::X
        test_oob<float, int32_t>(q, {5},       0,  0);                 // valid
        test_oob<float, int32_t>(q, {5},       0, +1);                 // overflow
        test_oob<float, int32_t>(q, {5},       0, -1, true,  true);    // underflow, negIdx=true
        test_oob<float, int32_t>(q, {5},       0, -1, true,  false);   // underflow, negIdx=false
        test_oob<float, int32_t>(q, {3,4,5},   2, +1);                 // overflow
        test_oob<float, int64_t>(q, {3,4,5},   1, -1, true,  true);    // underflow

        // Target::Y
        test_oob<float, int32_t>(q, {5,3},     0, +1, false);          // overflow
        test_oob<float, int32_t>(q, {3,5},     1, -1, false, true);    // underflow
        test_oob<float, int32_t>(q, {3,4,5},   1,  0, false);          // valid
    }
    catch (std::exception const& e) {
        std::cerr << "Failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}