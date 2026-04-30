#include "../fwd.cuh"

namespace sm86::prefill {

template void run_fwd_phase1_kernel<576, false>(const SparseAttnFwdParams&);
template void run_fwd_phase1_kernel<576, true>(const SparseAttnFwdParams&);

}
