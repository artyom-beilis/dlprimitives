///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/net.hpp>
namespace dlprim {
    ///
    /// Namespace that contains various optimizers
    ///
    namespace solvers {
        /// Base class for SGD based optimizers
        class SolverBase {
        public:
            /// Prepare solver - takes all parameters that need to be trained and prepares buffers
            virtual void init(Net &n,ExecutionContext const &q) = 0;
            /// zero all gradients before accumulating them for next batch
            virtual void zero_grad(Net &n,ExecutionContext const &e) = 0;
            /// apply solver updates 
            virtual void apply(Net &n,ExecutionContext const &e) = 0;
            virtual ~SolverBase() {}
            ///
            /// shortcut for single training step zero_grad, forward, backward, apply
            ///
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
