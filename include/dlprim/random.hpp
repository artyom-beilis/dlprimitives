///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cstdint>
#include <array>
#include <time.h>

namespace dlprim {

    /// CPU implementation
    /// of Philox 4-32-10 algorithm
    namespace philox {

        constexpr int result_items = 4;
        
        /// Each round returns 4 items - result 
        typedef std::array<std::uint32_t,result_items> uint_result_type;
        /// Each round returns 4 items - in range[0.0,1.0f)
        typedef std::array<float,result_items> float_result_type;

        ///
        /// Get float result in range[0.0,1.0f)
        ///
        float_result_type calculate_float(std::uint64_t seed, std::uint64_t seq);
        ///
        /// Get uint32 result in range[0::UINT_MAX]
        ///
        uint_result_type calculate_integer(std::uint64_t seed, std::uint64_t seq);
        
    };

    class RandomState {
    public:
        typedef std::uint64_t seed_type;
        typedef std::uint64_t sequence_type;
        RandomState(std::uint64_t seed)
        {
            seed_ = seed;
            sequence_ = 0;
        }
        RandomState()
        {
            seed_ = time(nullptr) + 0xDEADBEEF;
        }
        std::uint64_t sequence_bump(size_t items) 
        {
            std::uint64_t cur = sequence_;
            sequence_ += items;
            return cur; 
        }
        void seed(std::uint64_t s)
        {
            seed_ = s;
            sequence_ = 0;
        }
        std::uint64_t seed() const
        {
            return seed_;
        }
        
    private:

        std::uint64_t sequence_;
        std::uint64_t seed_;
    };


    
} // dlprim
