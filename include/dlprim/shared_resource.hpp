#pragma once
#include <dlprim/random.hpp>
namespace dlprim {
    ///
    /// Resources shared by the entire network
    ///
    /// Currently it includes:
    /// 
    /// - random number generator state
    /// 
    class SharedResource {
    public:
        /// State of random number generator
        RandomState &rng_state()
        {
            return state_;
        }
    private:
        RandomState state_;
    };

};
