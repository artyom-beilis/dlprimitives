///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>

#if defined(__WIN32) || defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__)
#   define  DLPRIM_WINDOWS
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
#	define DLPRIM_API  __attribute__((visibility("default")))
#endif     


///
/// Mane namespace
///
namespace dlprim {
    namespace json { class value; }

    ///
    /// Base dlprim excetion
    ///
    class Error : public std::runtime_error {
    public:
        Error(std::string const &v) : std::runtime_error(v) {}
    };

    ///
    /// Thrown if some stuff is not implemented yet
    ///
    class NotImplementedError : public Error {
    public:
        NotImplementedError(std::string const &v) : Error(v) {}
    };

    ///
    /// Thrown in case of invalid parameters
    ///
    class ValidationError : public Error {
    public:
        ValidationError(std::string const &v) : Error(v) {}
    };

    ///
    /// Thrown if OpenCL kernel compilation failed.
    ///
    class BuildError : public Error {
    public:
        BuildError(std::string const &msg,std::string const &log) : Error(msg), log_(log) {}
        /// get full build log
        std::string const &log() const 
        {
            return log_;
        }
    private:
        std::string log_;
    };

    #define DLPRIM_CHECK(x) \
    do { if(!(x)) throw ValidationError(std::string("Failed " #x " at " __FILE__ ":") + std::to_string(__LINE__) ); } while(0)

    /// type definition
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

    /// returns true of data is double, float, half or bfloat16 type
    inline bool is_floating_point_data_type(DataType d)
    {
        return (d >> 3) < 2;
    }

    template<typename T>
    struct TypeTraits;

    template<>
    struct TypeTraits<float> { static constexpr DataType data_type = float_data; };

    template<>
    struct TypeTraits<uint16_t> { static constexpr DataType data_type = uint16_data; };

    template<>
    struct TypeTraits<int16_t> { static constexpr DataType data_type = int16_data; };

    template<>
    struct TypeTraits<uint8_t> { static constexpr DataType data_type = uint8_data; };

    template<>
    struct TypeTraits<int8_t> { static constexpr DataType data_type = int8_data; };


    template<>
    struct TypeTraits<uint64_t> { static constexpr DataType data_type = uint64_data; };

    template<>
    struct TypeTraits<int64_t> { static constexpr DataType data_type = int64_data; };


    template<>
    struct TypeTraits<double> { static constexpr DataType data_type = double_data; };

    template<>
    struct TypeTraits<uint32_t> { static constexpr DataType data_type = int32_data; };

    template<>
    struct TypeTraits<int32_t> { static constexpr DataType data_type = int32_data; };
    
    inline DataType string_to_data_type(std::string const &s)
    {
        if(s == "float" || s=="float32")
            return float_data;
        else if(s == "float16" || s=="half")
            return half_data;
        else if(s == "bfloat16")
            return bfloat16_data;
        else if(s == "int32" || s == "int")
            return int32_data;
        else if(s == "int8")
            return int8_data;
        else if(s == "uint8")
            return uint8_data;
        else if(s == "int16")
            return int16_data;
        else if(s == "uint16")
            return uint16_data;
        else if(s == "int64")
            return int64_data;
        else if(s == "uint64")
            return uint64_data;
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
    enum DataTypeLimit {
        dt_min_val,
        dt_max_val,
    };

    inline std::string data_type_to_opencl_numeric_limit(DataType dt,DataTypeLimit lmt)
    {
        if(is_floating_point_data_type(dt)) {
            std::string prefix;
            switch(dt) {
            case float_data: 
            case bfloat16_data:
                prefix="FLT"; 
                break;
            case double_data : prefix="DBL"; break;
            case half_data: prefix="HALF"; break;
            default:
                throw ValidationError("Unsupported type");
            }
            switch(lmt) {
            case dt_min_val: return "(-" + prefix + "_MAX)";
            case dt_max_val: return prefix + "_MAX";
            };
        }
        else {
            bool unsig = (dt & (3 << 3)) == (3<<3);
            std::string prefix;
            switch(dt) {
            case int64_data:  prefix = "LONG"; break;
            case uint64_data: prefix = "ULONG"; break;

            case int32_data:  prefix = "INT"; break;
            case uint32_data: prefix = "UINT"; break;

            case int16_data:  prefix = "SHRT"; break;
            case uint16_data: prefix = "USHRT"; break;

            case int8_data:   prefix = "CHAR"; break;
            case uint8_data:  prefix = "UCHAR"; break;
            default:
                throw NotImplementedError("Unsupported data type");
            }
            switch(lmt) {
            case dt_min_val: return unsig ? "0" : prefix + "_MIN";
            case dt_max_val: return prefix + "_MAX";
            };
        }
        throw NotImplementedError("Unsupported data type");
    }
    inline std::string data_type_to_opencl_type(DataType dt,bool io_type=false,bool kernel_param = false)
    {
        switch(dt) {
        case double_data: return "double";
        case int64_data: return "long";
        case uint64_data: return "ulong";

        case float_data: return "float";
        case int32_data: return "int";
        case uint32_data: return "uint";

        case half_data: return (kernel_param ? "float" : "half");
        case bfloat16_data: return (io_type ? "ushort" : "float" );
        case int16_data: return "short";
        case uint16_data: return "ushort";

        case int8_data: return "char";
        case uint8_data: return "uchar";
        default:
            throw NotImplementedError("Unsupported data type");
        }
    }
    inline std::string data_type_to_opencl_param_type(DataType dt)
    {
        return data_type_to_opencl_type(dt,false,true);
    }

    constexpr int size_of_data_type(DataType d)
    {
        return 1 << ((d & 0x7) - 1);
    }

    /// Maximal number of dimensions in tensor
    static constexpr int max_tensor_dim = 8;

    /// internal flag
	constexpr int forward_data = 1;
    /// internal flag
	constexpr int backward_data = 2;
    /// internal flag
	constexpr int backward_param = 3;

	///
    /// Parameterless Activations that can be embedded to general kernels like inner product or convolution
    ///
    enum class StandardActivations : int {
		identity = 0,
		relu = 1,
        tanh = 2,
        sigmoid = 3,
        relu6 = 4,
	};
    
    StandardActivations activation_from_name(std::string const &name);
    char const *activation_to_name(StandardActivations act);
    std::string activation_equation(StandardActivations act,std::string const &variable); 
    std::string activation_backward_equation(StandardActivations act,std::string const &dy,std::string const &y);


    ///
    /// Operation mode of layers - inference of training
    ///
    enum class CalculationsMode {
        train,
        predict
    };
    

    /// 
    /// internal GEMM mode
    ///
    enum class GemmOpMode {
        forward = 1,
        backward_filter = 2,
        backward_data = 3
    };
	
    ///
    /// Convolution settings
    ///
    struct Convolution2DConfigBase {
		int channels_in = -1;
		int channels_out = -1;
		int kernel[2] = {1,1};
		int stride[2] = {1,1};
		int dilate[2] = {1,1};
		int pad[2] = {0,0};
		int groups = 1;
    };
    ///
    /// Interpolation methods
    ///
    enum class InterpolateType {
        nearest  = 0,
        nearest_exact  = 1,
        bilinear = 2
    };



}
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

