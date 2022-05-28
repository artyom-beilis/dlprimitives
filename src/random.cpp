///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/random.hpp>

namespace dlprim {
    namespace philox {
        namespace details {

            inline std::uint32_t mulhi(std::uint32_t a,std::uint32_t b)
            {
                std::uint64_t v=a;
                v*=b;
                return v>>32;
            }

            inline std::uint32_t mullo(std::uint32_t a,std::uint32_t b)
            {
                std::uint64_t v=a;
                v*=b;
                return v;
            }

            
            inline void mulhilo(std::uint32_t a,std::uint32_t b,std::uint32_t &h,std::uint32_t &l)
            {
                std::uint64_t hl = (std::uint64_t)(a)*(std::uint64_t)(b);
                h = hl >> 32;
                l = hl;
            }

            struct state {
                std::uint32_t l0,r0,l1,r1;
                std::uint32_t k0,k1;
            };

            inline state single_round(state s)
            {
                state next;
                next.l1 = mullo(s.r1, 0xD2511F53);
                next.r1 = mulhi(s.r0, 0xCD9E8D57) ^ s.k0 ^ s.l0;
                next.l0 = mullo(s.r0, 0xCD9E8D57);
                next.r0 = mullo(s.r1, 0xD2511F53) ^ s.k1 ^ s.l1;
                next.k0 = s.k0 + 0xBB67AE85;
                next.k1 = s.k1 + 0x9E3779B9;
                return next;
            }
            
            inline state make_initial_state(std::uint64_t seed,std::uint64_t sequence)
            {
                state s = state();
                s.l1 = sequence >> 32;
                s.r1 = sequence;
                s.k0 = seed;
                s.k1 = seed >> 32;
                return s;
            }

            inline uint_result_type calculate(state s)
            {
                for(int i=0;i<10;i++)
                    s=single_round(s);
                uint_result_type r;
                r[0] = s.l0;
                r[1] = s.r0;
                r[2] = s.l1;
                r[3] = s.r1;
                return r;
            }
            inline float_result_type calculate_float(state s)
            {
                uint_result_type r = calculate(s);
                float_result_type rf;
                /// make sure float does not become 1 after rounding
                /// 24 - for float/bfloat16
                /// 16 - for half
                /// 32 - for double
                constexpr int accuracy_shift = 24;
                constexpr int drop_bits = 32 - accuracy_shift;
                constexpr float factor = 1.0f / (std::uint64_t(1) << accuracy_shift);
                for(int i=0;i<4;i++)
                    rf[i] =(r[i] >> drop_bits) * factor;
                return rf;
            }
        }

        float_result_type calculate_float(std::uint64_t seed, std::uint64_t seq)
        {
            return details::calculate_float(details::make_initial_state(seed,seq));
        }
        
        uint_result_type calculate_integer(std::uint64_t seed, std::uint64_t seq)
        {
            return details::calculate(details::make_initial_state(seed,seq));
        }

    }
}
