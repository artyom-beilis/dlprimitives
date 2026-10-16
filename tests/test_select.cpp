#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/core/select.hpp>
#include "test.hpp"
#include <iostream>
#include <cstring>
#include <vector>
#include <cstdint>

namespace dp = dlprim;

static dp::Shape to_shape(std::vector<int64_t> const& v)
{
    std::vector<long> tmp(v.begin(), v.end());
    return dp::Shape::from_range(tmp.begin(), tmp.end());
}

static std::vector<int64_t> strides_of(std::vector<int64_t> const& shape)
{
    std::vector<int64_t> s(shape.size());
    int64_t acc = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        s[i] = acc;
        acc *= shape[i];
    }
    return s;
}

static int64_t numel(std::vector<int64_t> const& s)
{
    int64_t n = 1;
    for (auto x : s) n *= x;
    return n;
}

static std::vector<int64_t> unravel(int64_t i, std::vector<int64_t> const& shape)
{
    std::vector<int64_t> c(shape.size());
    for (int k = (int)shape.size() - 1; k >= 0; --k) {
        c[k] = i % shape[k];
        i /= shape[k];
    }
    return c;
}

static int64_t ravel(std::vector<int64_t> const& c, std::vector<int64_t> const& st)
{
    int64_t off = 0;
    for (size_t k = 0; k < c.size(); ++k) off += c[k] * st[k];
    return off;
}

template<typename T>
static dp::Tensor mk(dp::ExecutionContext const& q,
    std::vector<int64_t> shape,
    std::vector<T> const& v)
{
    dp::Context ctx(q);
    dp::Tensor t(ctx, to_shape(shape), dp::TypeTraits<T>::data_type);
    t.to_device(q, const_cast<T*>(v.data()));
    return t;
}

template<typename T>
static dp::Tensor zeros(dp::ExecutionContext const& q, std::vector<int64_t> shape)
{
    return mk<T>(q, shape, std::vector<T>(numel(shape), T(0)));
}

static bool eq(dp::Tensor a, dp::Tensor b, dp::ExecutionContext const& q)
{
    a.to_host(q); b.to_host(q);
    return a.shape() == b.shape() && a.dtype() == b.dtype()
        && memcmp(a.host_data(), b.host_data(), a.memory_size()) == 0;
}

static void run(dp::ExecutionContext const& q,
    dp::Tensor& x, dp::Tensor& y, dp::Tensor& idx,
    std::vector<int64_t> const& xst,
    std::vector<int64_t> const& yst,
    std::vector<int64_t> const& ist,
    uint64_t axis,
    dp::core::Select::Op op, dp::core::Select::Target tg,
    bool scal = false, double alpha = 0.0,
    dp::Tensor* flag = nullptr)
{
    dp::Context ctx(q);
    auto s = dp::core::Select::create(
        ctx, (int)xst.size(), x.dtype(), idx.dtype(), op, tg, scal);
    s->enqueue(x, y, idx, to_shape(xst), to_shape(yst), to_shape(ist),
        axis, flag, alpha, q);
}

// broadcast istrds: only axis is 1, others to 0
static std::vector<int64_t> bcast_strides(int n, int axis)
{
    std::vector<int64_t> s(n, 0);
    s[axis] = 1;
    return s;
}

template<typename D, typename I>
static void title(char const* name, std::vector<int64_t> const& shape, int axis)
{
    std::cout << name << ": dims=" << shape.size()
        << " shape=[";
    for (size_t i = 0; i < shape.size(); ++i)
        std::cout << (i ? "," : "") << shape[i];
    std::cout << "] axis=" << axis
        << " D=" << (std::is_same<D, float>::value ? "f32" :
            std::is_same<D, int32_t>::value ? "i32" : "?")
        << " I=" << (std::is_same<I, int32_t>::value ? "i32" : "i64")
        << std::endl;
}

template<typename D, typename I>
void test_index_select(dp::ExecutionContext const& q,
    std::vector<int64_t> xshape, int axis)
{
    title<D, I>("index_select", xshape, axis);

    const int64_t K = 2;
    auto yshape = xshape; yshape[axis] = K;

    std::vector<int64_t> index = { 0, xshape[axis] - 1 };

    auto xst = strides_of(xshape);
    auto yst = strides_of(yshape);
    auto ist = bcast_strides((int)xshape.size(), axis);

    auto xsize = numel(xshape);
    auto ysize = numel(yshape);

    std::vector<D> xd(xsize);
    for (int64_t i = 0; i < xsize; ++i) xd[i] = (D)(i + 1);

    // 1-D index is broadcast along `axis`; physical offsets 0..K-1 hold the index values themselves
    std::vector<I> idxd(ysize, I(0));
    for (int64_t k = 0; k < K; ++k) idxd[k] = (I)index[k];

    std::vector<D> yd(ysize);
    for (int64_t out = 0; out < ysize; ++out) {
        auto c = unravel(out, yshape);
        auto ic = c;
        ic[axis] = index[c[axis]];
        yd[out] = xd[ravel(ic, xst)];
    }

    auto x = mk<D>(q, xshape, xd);
    auto idx = mk<I>(q, yshape, idxd);
    auto y = zeros<D>(q, yshape);

    run(q, x, y, idx, xst, yst, ist, axis,
        dp::core::Select::ASSIGN, dp::core::Select::X);

    auto ey = mk<D>(q, yshape, yd);
    TEST(eq(y, ey, q));
}

template<typename D, typename I>
void test_index_select_alpha(dp::ExecutionContext const& q,
    std::vector<int64_t> xshape, int axis,
    double alpha)
{
    title<D, I>("index_select+alpha", xshape, axis);

    const int64_t K = 2;
    auto yshape = xshape; yshape[axis] = K;
    std::vector<int64_t> index = { 0, xshape[axis] - 1 };

    auto xst = strides_of(xshape);
    auto yst = strides_of(yshape);
    auto ist = bcast_strides((int)xshape.size(), axis);

    auto xsize = numel(xshape);
    auto ysize = numel(yshape);

    std::vector<D> xd(xsize);
    for (int64_t i = 0; i < xsize; ++i) xd[i] = (D)(i + 1);

    std::vector<I> idxd(ysize, I(0));
    for (int64_t k = 0; k < K; ++k) idxd[k] = (I)index[k];

    std::vector<D> yd(ysize);
    for (int64_t out = 0; out < ysize; ++out) {
        auto c = unravel(out, yshape);
        auto ic = c;
        ic[axis] = index[c[axis]];
        yd[out] = (D)(alpha * (double)xd[ravel(ic, xst)]);
    }

    auto x = mk<D>(q, xshape, xd);
    auto idx = mk<I>(q, yshape, idxd);
    auto y = zeros<D>(q, yshape);

    run(q, x, y, idx, xst, yst, ist, axis,
        dp::core::Select::ASSIGN, dp::core::Select::X,
        /*scal=*/true, alpha);

    auto ey = mk<D>(q, yshape, yd);
    TEST(eq(y, ey, q));
}

template<typename D, typename I>
void test_gather(dp::ExecutionContext const& q,
    std::vector<int64_t> xshape, int axis)
{
    title<D, I>("gather", xshape, axis);

    auto yshape = xshape;
    yshape[axis] = std::min<int64_t>(xshape[axis], 2);

    auto xst = strides_of(xshape);
    auto yst = strides_of(yshape);
    auto ist = strides_of(yshape);

    auto xsize = numel(xshape);
    auto ysize = numel(yshape);

    std::vector<D> xd(xsize);
    for (int64_t i = 0; i < xsize; ++i) xd[i] = (D)(i + 1);

    std::vector<I> idxd(ysize);
    std::vector<D> yd(ysize);
    for (int64_t out = 0; out < ysize; ++out) {
        auto c = unravel(out, yshape);
        int64_t iv = (c[axis] + 1) % xshape[axis];
        idxd[out] = (I)iv;
        auto ic = c;
        ic[axis] = iv;
        yd[out] = xd[ravel(ic, xst)];
    }

    auto x = mk<D>(q, xshape, xd);
    auto idx = mk<I>(q, yshape, idxd);
    auto y = zeros<D>(q, yshape);

    run(q, x, y, idx, xst, yst, ist, axis,
        dp::core::Select::ASSIGN, dp::core::Select::X);

    auto ey = mk<D>(q, yshape, yd);
    TEST(eq(y, ey, q));
}

template<typename D, typename I>
void test_index_add(dp::ExecutionContext const& q,
    std::vector<int64_t> selfshape, int axis,
    double alpha = 1.0)
{
    title<D, I>("index_add", selfshape, axis);

    const int64_t K = 2;
    auto sshape = selfshape; sshape[axis] = K;

    std::vector<int64_t> index = { 0, selfshape[axis] - 1 };

    auto selfst = strides_of(selfshape);
    auto sst = strides_of(sshape);
    auto ist = bcast_strides((int)selfshape.size(), axis);

    auto ssize = numel(sshape);
    auto ysize = numel(selfshape);

    std::vector<D> srcd(ssize);
    for (int64_t i = 0; i < ssize; ++i) srcd[i] = (D)(i + 1);

    std::vector<I> idxd(ssize, I(0));
    for (int64_t k = 0; k < K; ++k) idxd[k] = (I)index[k];

    std::vector<D> yd(ysize, D(0));
    for (int64_t out = 0; out < ssize; ++out) {
        auto c = unravel(out, sshape);
        auto oc = c;
        oc[axis] = index[c[axis]];
        yd[ravel(oc, selfst)] += (D)(alpha * (double)srcd[out]);
    }

    auto src = mk<D>(q, sshape, srcd);
    auto idx = mk<I>(q, sshape, idxd);
    auto y = zeros<D>(q, selfshape);

    run(q, src, y, idx, sst, selfst, ist, axis,
        dp::core::Select::ADD, dp::core::Select::Y,
        /*scal=*/true, alpha);

    auto ey = mk<D>(q, selfshape, yd);
    TEST(eq(y, ey, q));
}

template<typename D, typename I>
void test_scatter_add(dp::ExecutionContext const& q,
    std::vector<int64_t> selfshape, int axis)
{
    title<D, I>("scatter_add", selfshape, axis);

    auto srcshape = selfshape;
    auto indexshape = selfshape;

    auto selfst = strides_of(selfshape);
    auto sst = strides_of(srcshape);
    auto ist = strides_of(indexshape);

    auto ssize = numel(srcshape);
    auto ysize = numel(selfshape);

    std::vector<D> srcd(ssize);
    for (int64_t i = 0; i < ssize; ++i) srcd[i] = (D)(i + 1);

    std::vector<I> idxd(ssize);
    std::vector<D> yd(ysize, D(0));
    for (int64_t out = 0; out < ssize; ++out) {
        auto c = unravel(out, indexshape);
        int64_t iv = (c[axis] + 1) % selfshape[axis];
        idxd[out] = (I)iv;
        auto oc = c;
        oc[axis] = iv;
        yd[ravel(oc, selfst)] += srcd[out];
    }

    auto src = mk<D>(q, srcshape, srcd);
    auto idx = mk<I>(q, indexshape, idxd);
    auto y = zeros<D>(q, selfshape);

    run(q, src, y, idx, sst, selfst, ist, axis,
        dp::core::Select::ADD, dp::core::Select::Y);

    auto ey = mk<D>(q, selfshape, yd);
    TEST(eq(y, ey, q));
}

template<typename D, typename I>
void test_oob(dp::ExecutionContext const& q,
    std::vector<int64_t> shape, int axis, bool oob, bool is_x=true)
{
    char tit[13];
    sprintf(tit, "oob_%c(%s)", is_x ? 'X' : 'Y', oob ? "true" : "false");
    title<D, I>(tit, shape, axis);

    auto small_shape = shape; small_shape[axis] = 2;
    std::vector<int64_t> index = {
        0, oob ? (int64_t)shape[axis] : (int64_t)shape[axis] - 1
    };

    // X: src = shape, dst = small_shape, idx = small_shape
    // Y: src = small_shape,  dst = shape, idx = small_shape
    auto& xshape = is_x ? shape : small_shape;
    auto& yshape = is_x ? small_shape : shape;

    auto xst = strides_of(xshape);
    auto yst = strides_of(yshape);
    auto ist = bcast_strides((int)shape.size(), axis);

    auto xsize = numel(xshape);
    auto isize = numel(small_shape);

    std::vector<D> xd(xsize);
    for (int64_t i = 0; i < xsize; ++i) xd[i] = (D)(i + 1);

    std::vector<I> idxd(isize, I(0));
    idxd[0] = (I)index[0];
    idxd[1] = (I)index[1];

    auto x = mk<D>(q, xshape, xd);
    auto idx = mk<I>(q, small_shape, idxd);
    auto y = zeros<D>(q, yshape);
    auto flag = zeros<int32_t>(q, { 1 });

    run(q, x, y, idx, xst, yst, ist, axis,
        dp::core::Select::ASSIGN,
        is_x ? dp::core::Select::X : dp::core::Select::Y,
        /*scal=*/false, 0.0, &flag);

    flag.to_host(q);
    int32_t expected = oob ? -1 : 0;
    TEST(flag.data<int32_t>()[0] == expected);
}

int main(int argc, char** argv)
{
    if (argc != 2) { std::cerr << "Usage: platform:device\n"; return 1; }
    try {
        dp::Context ctx(argv[1]);
        if (ctx.is_cpu_context()) return 0;
        auto q = ctx.make_execution_context();
        std::cout << ctx.name() << std::endl;

        // -------- index_select: DIMS = 1..8, axis = 0 --------------
        test_index_select<float, int32_t>(q, { 5 }, 0);
        test_index_select<float, int32_t>(q, { 3,4 }, 0);
        test_index_select<float, int32_t>(q, { 3,2,4 }, 0);
        test_index_select<float, int32_t>(q, { 3,2,2,4 }, 0);
        test_index_select<float, int32_t>(q, { 3,2,2,2,4 }, 0);
        test_index_select<float, int32_t>(q, { 3,2,2,2,2,4 }, 0);
        test_index_select<float, int32_t>(q, { 3,2,2,2,2,2,4 }, 0);
        test_index_select<float, int32_t>(q, { 3,2,2,2,2,2,2,4 }, 0);

        // -------- index_select: axis scan ---------------------------
        test_index_select<float, int32_t>(q, { 3,4,5 }, 1);
        test_index_select<float, int32_t>(q, { 3,4,5 }, 2);
        test_index_select<float, int32_t>(q, { 3,4,5,6 }, 1);
        test_index_select<float, int32_t>(q, { 3,4,5,6 }, 2);
        test_index_select<float, int32_t>(q, { 3,4,5,6 }, 3);
        test_index_select<float, int32_t>(q, { 2,3,4,5,6 }, 2);
        test_index_select<float, int32_t>(q, { 2,3,4,5,6,7 }, 3);
        test_index_select<float, int32_t>(q, { 2,3,4,5,6,7 }, 5);

        // -------- index_select: int64 index --------------------------
        test_index_select<float, int64_t>(q, { 3,4 }, 0);
        test_index_select<float, int64_t>(q, { 3,4,5 }, 0);
        test_index_select<float, int64_t>(q, { 3,4,5 }, 1);
        test_index_select<float, int64_t>(q, { 3,4,5 }, 2);
        test_index_select<float, int64_t>(q, { 3,4,5,6 }, 2);

        // -------- index_select: int32 data ---------------------------
        test_index_select<int32_t, int32_t>(q, { 3,4,5 }, 0);
        test_index_select<int32_t, int32_t>(q, { 3,4,5 }, 1);
        test_index_select<int32_t, int64_t>(q, { 3,4,5 }, 1);
        test_index_select<int32_t, int64_t>(q, { 3,4,5,6 }, 2);

        // -------- index_select + alpha -------------------------------
        test_index_select_alpha<float, int32_t>(q, { 3,4 }, 0, 2.0);
        test_index_select_alpha<float, int32_t>(q, { 3,4,5 }, 1, 0.5);
        test_index_select_alpha<float, int64_t>(q, { 3,4,5 }, 2, -1.0);
        test_index_select_alpha<float, int32_t>(q, { 3,4,5,6 }, 0, 3.0);

        // -------- gather ---------------------
        test_gather<float, int32_t>(q, { 3,4 }, 0);
        test_gather<float, int32_t>(q, { 3,4 }, 1);
        test_gather<float, int32_t>(q, { 3,4,5 }, 0);
        test_gather<float, int32_t>(q, { 3,4,5 }, 1);
        test_gather<float, int32_t>(q, { 3,4,5 }, 2);
        test_gather<float, int32_t>(q, { 3,4,5,6 }, 1);
        test_gather<float, int32_t>(q, { 3,4,5,6 }, 3);
        test_gather<float, int64_t>(q, { 3,4,5 }, 1);
        test_gather<int32_t, int32_t>(q, { 3,4,5 }, 2);

        // -------- index_add -----------------------------------------
        test_index_add<float, int32_t>(q, { 3,4 }, 0);
        test_index_add<float, int32_t>(q, { 3,4 }, 1);
        test_index_add<float, int32_t>(q, { 3,4,5 }, 0);
        test_index_add<float, int32_t>(q, { 3,4,5 }, 1);
        test_index_add<float, int32_t>(q, { 3,4,5 }, 2);
        test_index_add<float, int32_t>(q, { 3,4,5,6 }, 0);
        test_index_add<float, int32_t>(q, { 3,4,5,6 }, 2);
        test_index_add<float, int32_t>(q, { 3,4,5,6,7 }, 3);
        test_index_add<float, int32_t>(q, { 2,3,4,5,6 }, 1);
        test_index_add<float, int64_t>(q, { 3,4,5 }, 1);
        test_index_add<float, int64_t>(q, { 3,4,5,6 }, 2);
        test_index_add<float, int32_t>(q, { 3,4,5 }, 1, 0.25);
        test_index_add<float, int32_t>(q, { 3,4,5 }, 2, 2.0);

        // -------- scatter_add ---------------------------------------
        test_scatter_add<float, int32_t>(q, { 3,4 }, 0);
        test_scatter_add<float, int32_t>(q, { 3,4 }, 1);
        test_scatter_add<float, int32_t>(q, { 3,4,5 }, 0);
        test_scatter_add<float, int32_t>(q, { 3,4,5 }, 1);
        test_scatter_add<float, int32_t>(q, { 3,4,5 }, 2);
        test_scatter_add<float, int32_t>(q, { 3,4,5,6 }, 0);
        test_scatter_add<float, int32_t>(q, { 3,4,5,6 }, 3);
        test_scatter_add<float, int32_t>(q, { 2,3,4,5,6 }, 2);
        test_scatter_add<float, int64_t>(q, { 3,4,5 }, 1);
        test_scatter_add<int32_t, int32_t>(q, { 3,4,5 }, 1);

        // -------- out-of-bound flag ---------------------------------
        // Target::X
        test_oob<float, int32_t>(q, { 5 }, 0, true);
        test_oob<float, int32_t>(q, { 5 }, 0, false);
        test_oob<float, int32_t>(q, { 3,4 }, 1, true);
        test_oob<float, int32_t>(q, { 3,4,5 }, 2, true);
        test_oob<float, int32_t>(q, { 3,4,5,6 }, 2, true);
        test_oob<float, int32_t>(q, { 2,3,4,5,6 }, 1, true);
        test_oob<float, int64_t>(q, { 3,4,5 }, 1, true);
        test_oob<int32_t, int32_t>(q, { 3,4,5 }, 2, true);

        // Target::Y
        test_oob<float, int32_t>(q, { 5,3 }, 0, true, false);
        test_oob<float, int32_t>(q, { 3,5 }, 1, true, false);
        test_oob<float, int32_t>(q, { 3,4,5 }, 1, true, false);
        test_oob<float, int32_t>(q, { 3,4,5,6 }, 2, true, false);
        test_oob<float, int32_t>(q, { 2,3,4,5,6 }, 2, true, false);
        test_oob<float, int64_t>(q, { 3,4,5 }, 1, true, false);
        test_oob<int32_t, int32_t>(q, { 3,4,5 }, 2, true, false);
        test_oob<float, int32_t>(q, { 3,4,5 }, 1, false, false);
    }
    catch (std::exception const& e) {
        std::cerr << "Failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}