///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/json.hpp>
#include <dlprim/definitions.hpp>

namespace dlprim {
namespace utils {
    template<typename T,size_t S>
    T parse_enum(json::value const &v,std::string const &name,char const *(&names)[S],T def)
    {
        if(v.find(name).is_undefined())
            return def;
        std::string val = v.get<std::string>(name);
        for(size_t i=0;i<S;i++) {
            if(val == names[i])
                return static_cast<T>(i);
        }
        throw ValidationError("Invalid value " + val + " for filed " + name);
    }
    template<typename T,size_t S>
    void get_1dNd_from_json(json::value const &v,std::string const &name,T (&vals)[S],bool required=false)
    {
        json::value const &tmp = v.find(name);
        if(tmp.is_undefined()) {
            if(required)
                throw ValidationError("Missing value in json " + name);
            return;
        }
        else if(tmp.type()==json::is_number) {
            T val = tmp.get_value<T>();
            for(size_t i=0;i<S;i++)
                vals[i] = val;
        }
        else if(tmp.type()==json::is_array) {
            auto ar = tmp.get_value<std::vector<T> >();
            if(ar.size() != S)
                throw ValidationError("Array size of filed " + name + " must be " + std::to_string(S));
            for(size_t i=0;i<S;i++)
                vals[i] = ar[i];
        }
        else {
            throw ValidationError("Invalid filed value for " + name);
        }
    }
    inline StandardActivations activation_from_json(json::value const &v)
    {
        return activation_from_name(v.get("activation","identity"));
    }

} // util
} // json
