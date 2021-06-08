#pragma once
#include <stdexcept>

#if defined(__WIN32) || defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__)
#	if defined(DLL_EXPORT)
#		if defined(DLPRIM_SOURCE)
#			define DLPRIM_API __declspec(dllexport)
#		else
#			define DLPRIM_API __declspec(dllimport)
#		endif
#	else
#		define DLPRIM_API
#	endif
#else // ELF BINARIES
#	define DLPRIM_API
#endif


namespace dlprim {
    namespace json { class value; }

    class Error : public std::runtime_error {
    public:
        Error(std::string const &v) : std::runtime_error(v) {}
    };

    class NotImplementedError : public Error {
    public:
        NotImplementedError(std::string const &v) : Error(v) {}
    };

    class ValidationError : public Error {
    public:
        ValidationError(std::string const &v) : Error(v) {}
    };

    class BuildError : public Error {
    public:
        BuildError(std::string const &msg,std::string const &log) : Error(msg), log_(log) {}
        std::string const &log() const 
        {
            return log_;
        }
    private:
        std::string log_;
    };

    #define DLPRIM_CHECK(x) \
    do { if(!(x)) throw ValidationError(std::string("Failed " #x " at " __FILE__ ":") + std::to_string(__LINE__) ); } while(0)

    enum DataType {
        double_data   = 4 + (0 << 3),
        int64_data    = 4 + (2 << 3),
        uint64_data   = 4 + (3 << 3),

        float_data    = 3 + (0 << 3),
        int32_data    = 3 + (2 << 3),
        uint32_data   = 3 + (3 << 3),

        half_data     = 2 + (0 << 3),
        bfloat16_data = 2 + (1 << 3),
        int16_data    = 2 + (2 << 3),
        uint16_data   = 2 + (3 << 3),

        int8_data     = 1 + (2 << 3),
        uint8_data    = 1 + (3 << 3),
    };

    template<typename T>
    struct TypeTraits;

    template<>
    struct TypeTraits<float> { static constexpr DataType data_type = float_data; };
    
    inline DataType string_to_data_type(std::string const &s)
    {
        if(s == "float" || s=="float32")
            return float_data;
        else if(s == "float16" || s=="half")
            return half_data;
        else if(s == "bfloat16")
            return bfloat16_data;
        throw ValidationError("Unknown data type " + s);
    }
    inline std::string data_type_to_string(DataType dt)
    {
        switch(dt) {
        case double_data: return "double";
        case int64_data: return "int64";
        case uint64_data: return "uint64";

        case float_data: return "float";
        case int32_data: return "int32";
        case uint32_data: return "uint32";

        case half_data: return "half";
        case bfloat16_data: return "bfloat16";
        case int16_data: return "int16";
        case uint16_data: return "uint16";

        case int8_data: return "int8";
        case uint8_data: return "uint8";
        default:
            return "unknown";
        }
    
    }

    constexpr int size_of_data_type(DataType d)
    {
        return 1 << ((d & 0x7) - 1);
    }

    static constexpr int max_tensor_dim = 4;
	constexpr int forward_data = 1;
	constexpr int backward_data = 2;
	constexpr int backward_param = 3;

	enum class StandardActivations : int {
		identity = 0,
		relu = 1,
        tanh = 2,
        sigmoid = 3,
        relu6 = 4,
	};
    
    StandardActivations activation_from_name(std::string const &name);
    char const *activation_to_name(StandardActivations act);


    enum class CalculationsMode {
        train,
        predict
    };


}
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

