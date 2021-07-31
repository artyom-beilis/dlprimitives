#pragma once
#include <dlprim/net.hpp>
namespace dlprim {
    namespace solvers {
        class SolverBase {
        public:
            virtual void init(Net &n,ExecutionContext const &q) = 0;
            virtual void zero_grad(Net &n,ExecutionContext const &e) = 0;
            virtual void apply(Net &n,ExecutionContext const &e) = 0;
            virtual ~SolverBase() {}
            void step(Net &n,ExecutionContext const &e)
            {
                zero_grad(n,e);
                n.forward(e);
                n.backward(e);
                apply(n,e);
            }
        };
    } // solvers
} // dlprim
