///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once 
#include <dlprim/tensor.hpp>

namespace dlprim {
    namespace json {
        class value;
    }
    ///
    /// Base class used for loading non-native model formats to dlprimitives
    ///
    class ModelBase {
    public:
        
        virtual  ~ModelBase(){}
        /// Return representation of the network
        virtual json::value const &network()  // =0 - due to boost python not fully abstractr
        {
            throw ValidationError("Implemente Me");
        }
        /// Return CPU tensor containing parameter value by give name 
        virtual Tensor get_parameter(std::string const &name) // =0 - due to boost python not fully abstractr
        {
            return Tensor();
        }
    };
}
