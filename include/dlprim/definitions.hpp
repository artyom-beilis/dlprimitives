#pragma once
#include <stdexcept>
namespace dlprim {
    class Error : public std::runtime_error {
    public:
        Error(std::string const &v) : std::runtime_error(v) {}
    };

    class NotImplementedError : public Error {
    public:
        NotImplementedError(std::string const &v) : Error(v) {}
    };

    class ValidatioError : public Error {
    public:
        ValidatioError(std::string const &v) : Error(v) {}
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
    do { if(!(x)) throw ValidatioError(std::string("Failed " #x " at " __FILE__ ) + std::to_string(__LINE__) ); } while(0)

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
	};

    enum class CalculationsMode {
        train,
        predict
    };


}
/// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

