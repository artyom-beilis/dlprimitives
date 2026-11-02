#include <dlprim/core/select.hpp>
#include <dlprim/gpu/program_cache.hpp>

namespace dlprim {
namespace core {

void bind_shape(cl::Kernel& k, int& p, Shape const& s);
cl::NDRange get_broadcast_ndrange(const Shape& ref);
void bind_as_dtype(cl::Kernel& k, int& p, double value, DataType dt);

Shape to_strides(const Shape &j) {
    Shape sp = j;
    size_t strd = 1;
    for (int i = j.size() - 1; i >= 0; i--) {
        sp[i] = strd;
        strd *= j[i];
    }
    return sp;
}

class SelectImpl :public Select {
public:
    SelectImpl(Context &ctx, int dims, DataType dtype, DataType itype, Op op, Target target, bool scal)
            : dims(dims), scal(scal), dtype_(op == Op::ASSIGN && !scal ? as_integer_type(dtype) : dtype),
              target(target) {
        DLPRIM_CHECK(!is_floating_point_data_type(itype));
        DLPRIM_CHECK(dims <= max_tensor_dim);
        cl::Program prg = gpu::Cache::instance().get_program(ctx, "select",
            "DIMS", dims,
            "dtype", data_type_to_opencl_type(dtype_),
            "itype", data_type_to_opencl_type(itype),
            "META", ((dtype_==float_data) << 2) | (op << 1) | scal,
            "#SELOP(x,y)",op == Op::ADD ? "atomic_add(&x,y)" : "x=y",
            "#RPOS", target == Target::X ? "_POS(y)" : "_POS(x)",
            "#SPOS", target == Target::X ? "_POS(x)" : "_POS(y)");
        sel_ = cl::Kernel(prg, "sel");
    }

    void enqueue(Tensor &x, Tensor &y, Tensor &idx, int64_t axis, Tensor *flag, double alpha, ExecutionContext e, bool negIdx) override {
        auto &sx = x.shape();
        DLPRIM_CHECK(sx.size() == dims);
        auto &sy = y.shape();
        DLPRIM_CHECK(sy.size() == dims);
        auto si = idx.shape();
        DLPRIM_CHECK(si.size() == 1 || si.size() == dims);
        Shape sti;
        if (si.size() == 1) {
            std::vector<size_t> temp(dims, 0);
            temp[axis] = 1;
            sti = Shape::from_range(temp.begin(), temp.end());
            size_t siz = si[0];
            si = sy;
            si[axis] = siz;
        } else {
            sti = to_strides(si);
        }
        cl::Kernel& sel = sel_;
        int p = 0;
        bind_shape(sel, p, si);
        x.set_arg(sel,p);
        idx.set_arg(sel,p);
        y.set_arg(sel, p);
        bind_shape(sel, p, to_strides(sx));
        bind_shape(sel, p, sti);
        bind_shape(sel, p, to_strides(sy));
        cl_long bound = (target == Y ? sy : sx)[axis];
        sel.setArg(p++, bound);
        sel.setArg(p++, negIdx * bound);
        sel.setArg(p++, axis);
        if (scal) {
            bind_as_dtype(sel, p, alpha, dtype_);
        }
        if (flag) {
            flag->set_arg(sel, p);
        } else {
            sel.setArg(p++, x.device_buffer());
            sel.setArg(p, CL_ULONG_MAX);
        }
        e.queue().enqueueNDRangeKernel(sel, cl::NullRange, get_broadcast_ndrange(si), cl::NullRange, e.events(), e.event("select"));
    }
private:
    const int dims, scal;
    const DataType dtype_;
    const Target target;
    cl::Kernel sel_;
};
std::unique_ptr<Select> Select::create(Context &ctx, int dims, DataType dtype, DataType itype, Op op, Target target, bool scal)
{
    std::unique_ptr<Select> ptr(new SelectImpl(ctx, dims, dtype, itype, op, target, scal));
    return ptr;
}
}
}
