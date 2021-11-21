#include <dlprim/core/conv.hpp>

namespace dlprim {
namespace core {

    bool is_fwd_onednn_compatible(Context &ctx,Conv2DSettings const &config,StandardActivations activation);
    std::unique_ptr<Conv2DForward> fwd_onednn_create(Context &ctx,Conv2DSettings const &config,bool bias,StandardActivations activation);

} 
}
