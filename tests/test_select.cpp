#include <dlprim/context.hpp>
#include <dlprim/tensor.hpp>
#include <dlprim/core/select.hpp>
#include "test.hpp"
#include <iostream>
#include <cstring>
#include <vector>
#include <cstdint>

#define TP(T) dp::TypeTraits<T>::data_type

namespace dp = dlprim;

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

static void run(dp::ExecutionContext const& q,
                dp::Tensor& x, dp::Tensor& y, dp::Tensor& idx,
                uint64_t axis,
                dp::core::Select::Op op, dp::core::Select::Target tg,
                bool scal = false, double alpha = 0.0,
                dp::Tensor* flag = nullptr)
{
    dp::Context ctx(q);
    auto s = dp::core::Select::create(
        ctx, (int)x.shape().size(), x.dtype(), idx.dtype(), op, tg, scal);
    s->enqueue(x, y, idx, axis, flag, alpha, q);
}

template<typename D, typename I>
static void title(char const* name, dp::Shape const& shape, int axis)
{
    std::cout << name << ": dims=" << shape.size()
              << " shape=" << shape
              << " axis=" << axis
              << " D=" << (std::is_same<D,float>::value ? "f32" : "i32")
              << " I=" << (std::is_same<I,int32_t>::value ? "i32" : "i64")
              << std::endl;
}

template<typename D, typename I>
void test_index_select(dp::ExecutionContext const& q,
                       dp::Shape xshape, int axis)
{
    title<D, I>("index_select", xshape, axis);

    const size_t K = 2;
    auto yshape = xshape; yshape[axis] = K;
    dp::Shape index = { 0, xshape[axis] - 1 };

    auto xsize = xshape.total_size();
    auto ysize = yshape.total_size();

    std::vector<D> xd(xsize);
    for (size_t i = 0; i < xsize; ++i) xd[i] = (D)(i + 1);

    std::vector<I> idxd(K);
    for (size_t k = 0; k < K; ++k) idxd[k] = (I)index[k];

    std::vector<D> yd(ysize);
    for (size_t out = 0; out < ysize; ++out) {
        auto c = unravel(out, yshape);
        auto ic = c;
        ic[axis] = index[c[axis]];
        yd[out] = xd[flat(ic, xshape)];
    }

    auto x   = mk(q, xshape, TP(D), xd.data());
    auto idx = mk(q, {K}, TP(I), idxd.data());
    auto y   = zeros(q, yshape, TP(D));

    run(q, x, y, idx, axis, dp::core::Select::ASSIGN, dp::core::Select::X);

    auto ey = mk(q, yshape, TP(D), yd.data());
    TEST(eq(y, ey, q));
}

template<typename D, typename I>
void test_index_select_alpha(dp::ExecutionContext const& q,
                             dp::Shape xshape, int axis, double alpha)
{
    title<D, I>("index_select+alpha", xshape, axis);

    const size_t K = 2;
    auto yshape = xshape; yshape[axis] = K;
    dp::Shape index = { 0, xshape[axis] - 1 };

    auto xsize = xshape.total_size();
    auto ysize = yshape.total_size();

    std::vector<D> xd(xsize);
    for (size_t i = 0; i < xsize; ++i) xd[i] = (D)(i + 1);

    std::vector<I> idxd(K);
    for (size_t k = 0; k < K; ++k) idxd[k] = (I)index[k];

    std::vector<D> yd(ysize);
    for (size_t out = 0; out < ysize; ++out) {
        auto c = unravel(out, yshape);
        auto ic = c;
        ic[axis] = index[c[axis]];
        yd[out] = (D)(alpha * (double)xd[flat(ic, xshape)]);
    }

    auto x   = mk(q, xshape, TP(D), xd.data());
    auto idx = mk(q, {K}, TP(I), idxd.data());
    auto y   = zeros(q, yshape, TP(D));

    run(q, x, y, idx, axis, dp::core::Select::ASSIGN, dp::core::Select::X,
        /*scal=*/true, alpha);

    auto ey = mk(q, yshape, TP(D), yd.data());
    TEST(eq(y, ey, q));
}

template<typename D, typename I>
void test_gather(dp::ExecutionContext const& q,
                 dp::Shape xshape, int axis)
{
    title<D, I>("gather", xshape, axis);

    auto yshape = xshape;
    yshape[axis] = std::min<size_t>(xshape[axis], 2);

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

    auto x = mk(q, xshape, TP(D), xd.data());
    auto idx = mk(q, yshape, TP(I), idxd.data());
    auto y = zeros(q, yshape, TP(D));

    run(q, x, y, idx, axis, dp::core::Select::ASSIGN, dp::core::Select::X);

    auto ey = mk(q, yshape, TP(D), yd.data());
    TEST(eq(y, ey, q));
}

template<typename D, typename I>
void test_index_add(dp::ExecutionContext const& q,
                    dp::Shape selfshape, int axis, double alpha = 1.0)
{
    title<D, I>("index_add", selfshape, axis);

    const size_t K = 2;
    auto sshape = selfshape; sshape[axis] = K;
    dp::Shape index = { 0, selfshape[axis] - 1 };

    auto ssize = sshape.total_size();
    auto ysize = selfshape.total_size();

    std::vector<D> srcd(ssize);
    for (size_t i = 0; i < ssize; ++i) srcd[i] = (D)(i + 1);

    std::vector<I> idxd(K);
    for (size_t k = 0; k < K; ++k) idxd[k] = (I)index[k];

    std::vector<D> yd(ysize, D(0));
    for (size_t out = 0; out < ssize; ++out) {
        auto c = unravel(out, sshape);
        auto oc = c;
        oc[axis] = index[c[axis]];
        yd[flat(oc, selfshape)] += (D)(alpha * (double)srcd[out]);
    }

    auto src = mk(q, sshape, TP(D), srcd.data());
    auto idx = mk(q, {K}, TP(I), idxd.data());
    auto y   = zeros(q, selfshape, TP(D));

    run(q, src, y, idx, axis, dp::core::Select::ADD, dp::core::Select::Y,
        /*scal=*/true, alpha);

    auto ey = mk(q, selfshape, TP(D), yd.data());
    TEST(eq(y, ey, q));
}

template<typename D, typename I>
void test_scatter_add(dp::ExecutionContext const& q,
                      dp::Shape selfshape, int axis)
{
    title<D, I>("scatter_add", selfshape, axis);

    auto sshape = selfshape;
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
    auto y = zeros(q, selfshape, TP(D));

    run(q, src, y, idx, axis, dp::core::Select::ADD, dp::core::Select::Y);

    auto ey = mk(q, selfshape, TP(D), yd.data());
    TEST(eq(y, ey, q));
}

template<typename D, typename I>
void test_oob(dp::ExecutionContext const& q,
    dp::Shape shape, int axis, bool oob, bool is_x=true)
{
    char tit[13];
    sprintf(tit, "oob_%c(%s)", is_x ? 'X' : 'Y', oob ? "true" : "false");
    title<D, I>(tit, shape, axis);

    auto small_shape = shape; small_shape[axis] = 2;
    dp::Shape index = {
        0, oob ? shape[axis] : shape[axis] - 1
    };

    auto& xshape = is_x ? shape : small_shape;
    auto& yshape = is_x ? small_shape : shape;

    auto xsize = xshape.total_size();
    auto isize = small_shape.total_size();

    std::vector<D> xd(xsize);
    for (size_t i = 0; i < xsize; ++i) xd[i] = (D)(i + 1);

    std::vector<I> idxd(isize, I(0));
    idxd[0] = (I)index[0];
    idxd[1] = (I)index[1];

    auto x = mk(q, xshape, TP(D), xd.data());
    auto idx = mk(q, small_shape, TP(I), idxd.data());
    auto y = zeros(q, yshape, TP(D));
    auto flag = zeros(q, { 1 }, TP(int32_t));

    run(q, x, y, idx, axis,
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