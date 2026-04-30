#include <pybind11/pybind11.h>

#include "sparse_fwd.h"
#ifdef FLASH_MLA_HAS_SM90
#include "sparse_decode.h"
#endif
#include "sm86/decode/sparse_fp8/sparse_decode.h"
#include "dense_decode.h"
#ifdef FLASH_MLA_HAS_SM100
#include "dense_fwd.h"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("sparse_decode_fwd", &sm86_sparse_attn_decode_interface);
#ifdef FLASH_MLA_HAS_SM90
    // m.def("sparse_decode_fwd", &sparse_attn_decode_interface); // SM90 version
#endif
#ifdef FLASH_MLA_HAS_SM100
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
#endif
}
