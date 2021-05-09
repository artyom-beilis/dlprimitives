#pragma once
namespace dlprim {
    namespace gpu {
        char const *activation_function(DataType dtype,StandardActivations act)
        {
            switch(act) {
            case StandardActivations.identity:
                return "#define ACTIVATION(x) (x)\n";
            case StandardActivations.relu:
                switch(dtype) {
                case float_data:
                    return "#define ACTIVATION(x) max(x,0);\n"
                default:
                    ;
                }
                break;
            }
            DLPRIM_CHECK(!"Unsupported activation");
        }

        void activation_in_place(Tesor &tensor,DataType dtype,StandardActivations act,cl::CommandQueue &q,cl::Event *e);
    }
}
