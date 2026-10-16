#include <dlprim/core/select.hpp>

namespace dlprim {
namespace core {
	void bind_shape(cl::Kernel& k, int& p, Shape const& s);
	cl::NDRange get_broadcast_ndrange(Shape ref);
	void bind_as_dtype(cl::Kernel& k, int& p, double value, DataType dt);

class SelectImpl :public Select {
public:
	SelectImpl(Context &ctx, int dims, DataType dtype, DataType itype, Op op, Target target, bool scal)
		:dims(dims), dtype(dtype), scal(scal), target(target)
	{
		DLPRIM_CHECK(!is_floating_point_data_type(itype));
		DLPRIM_CHECK(dims <= 8);
		cl::Program prg = gpu::Cache::instance().get_program(ctx, "select",
												"DIMS", dims,
												"dtype", data_type_to_opencl_type(dtype),
												"itype", data_type_to_opencl_type(itype),
												"META", ((dtype==float_data)<<3)|(op<<2)|(target<<1)|scal
												);
		sel_ = cl::Kernel(prg, "sel");
	}

	void enqueue(Tensor& x, Tensor& y, Tensor& idx, Shape xstrds, Shape ystrds, Shape istrds, uint64_t axis, Tensor *flag, double alpha,ExecutionContext e) override
	{
		DLPRIM_CHECK(xstrds.size() == dims);
		DLPRIM_CHECK(ystrds.size() == dims);
		DLPRIM_CHECK(istrds.size() == dims);
		cl::Kernel& sel = sel_;
		int p = 0;
		Shape limit = idx.shape();
		bind_shape(sel, p, limit);
		x.set_arg(sel,p);
		idx.set_arg(sel,p);
		y.set_arg(sel, p);
		bind_shape(sel, p, xstrds);
		bind_shape(sel, p, istrds);
		bind_shape(sel, p, ystrds);
		sel.setArg(p++, (uint64_t)(target == Target::Y ? y.shape() : x.shape())[axis]);
		sel.setArg(p++, axis);
		if (scal) {
			bind_as_dtype(sel, p, alpha, dtype);
		}
		if (flag) {
			flag->set_arg(sel, p);
		} else {
			sel.setArg(p++, nullptr);
			sel.setArg(p++, 0llu);
		}
		e.queue().enqueueNDRangeKernel(sel, cl::NullRange, get_broadcast_ndrange(limit), cl::NullRange, e.events(), e.event("select"));
	}
private:
	const int dims, scal;
	const Target target;
	const DataType dtype;
	cl::Kernel sel_;
};
std::unique_ptr<Select> Select::create(Context &ctx, int dims, DataType dtype, DataType itype, Op op, Target target, bool scal)
{
	std::unique_ptr<Select> ptr = std::make_unique<SelectImpl>(ctx, dims, dtype, itype, op, target, scal);
	return ptr;
}
}
}