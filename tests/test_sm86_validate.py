"""
SM86 FlashMLA Validation Test
===============================
Runs the RTX 3090 SM86 kernel against the original FlashMLA PyTorch reference.

The reference implements the EXACT math that the Hopper (SM90) FlashMLA kernel computes.
All data generation and reference computation follows test_flash_mla_dense_decoding.py.
The 3090 runs the SM86 kernel path; the reference runs anywhere (CPU or GPU).

Usage:
    python tests/test_sm86_validate.py           # full suite
    python tests/test_sm86_validate.py --quick   # edge cases only
    python tests/test_sm86_validate.py --verbose
"""

import argparse
import math
import random
import sys
from typing import Tuple

import torch

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests'))
import kernelkit as kk
import flash_mla


# ═══════════════════════════════════════════════════════════════
# Reference implementation (from test_flash_mla_dense_decoding.py)
# This is the GROUND TRUTH — same math as the Hopper kernel.
# ═══════════════════════════════════════════════════════════════

def reference_torch(
    cache_seqlens: torch.Tensor,    # [batch_size]
    block_table: torch.Tensor,      # [batch_size, ?]
    q: torch.Tensor,                # [batch_size, s_q, h_q, d]
    blocked_k: torch.Tensor,        # [?, block_size, h_kv, d]
    dv: int,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ground-truth PyTorch reference — same math as Hopper FlashMLA."""

    def _attention(batch_idx, query, kv, dv, is_causal):
        h_q = query.size(0)
        h_kv = kv.size(0)
        s_q = query.shape[-2]
        s_k = kv.shape[-2]
        device = query.device
        query = query.float()
        kv = kv.float()
        if h_kv != 1:
            kv = kv.repeat_interleave(h_q // h_kv, dim=0)
        kv[kv != kv] = 0.0
        attn_weight = query @ kv.transpose(-2, -1)
        if is_causal and query.size(1) > 1:
            mask = torch.ones(s_q, s_k, dtype=torch.bool, device=device)
            mask = mask.tril(diagonal=s_k - s_q)
            attn_bias = torch.zeros(s_q, s_k, dtype=torch.float, device=device)
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            attn_weight += attn_bias
        attn_weight /= math.sqrt(query.size(-1))
        lse = attn_weight.logsumexp(dim=-1)
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        output = attn_weight @ kv[..., :dv]
        lonely_q_mask = (lse == float("-inf"))
        output[lonely_q_mask.unsqueeze(-1).broadcast_to(h_q, s_q, dv)] = 0.0
        lse[lonely_q_mask] = float("+inf")
        return output, lse

    b, s_q, h_q, d = q.size()
    block_size = blocked_k.size(1)
    h_kv = blocked_k.size(2)
    cache_seqlens_cpu = cache_seqlens.cpu()
    out_ref = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
    lse_ref = torch.empty(b, h_q, s_q, dtype=torch.float32)
    for i in range(b):
        cur_len = int(cache_seqlens_cpu[i].item())
        cur_num_blocks = kk.cdiv(cur_len, block_size)
        cur_block_indices = block_table[i][0: cur_num_blocks]
        cur_kv = blocked_k[cur_block_indices].view(-1, h_kv, d)[:cur_len, ...]
        cur_out, cur_lse = _attention(
            i, q[i].transpose(0, 1), cur_kv.transpose(0, 1), dv, is_causal
        )
        out_ref[i] = cur_out.transpose(0, 1)
        lse_ref[i] = cur_lse
    out_ref = out_ref.to(q.dtype)
    return out_ref, lse_ref


# ═══════════════════════════════════════════════════════════════
# Test data generation (from test_flash_mla_dense_decoding.py)
# ═══════════════════════════════════════════════════════════════

def generate_test_data(B, sq, sk, hq, hkv, d, dv, pbs, varlen, zero_len, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cache_seqlens_cpu = torch.full((B,), sk, dtype=torch.int32, device='cpu')
    if varlen:
        for i in range(B):
            cache_seqlens_cpu[i] = max(int(random.normalvariate(sk, sk / 2)), sq)
    if zero_len:
        mask = torch.rand(B) > 0.5
        cache_seqlens_cpu[mask] = 0

    max_seqlen = max(int(cache_seqlens_cpu.max().item()), 64)
    max_seqlen_pad = kk.cdiv(max_seqlen, 256) * 256

    q = torch.randn(B, sq, hq, d, dtype=torch.bfloat16, device='cuda') / 10
    q.clamp_(min=-1.0, max=1.0)

    bt = torch.arange(B * max_seqlen_pad // pbs, dtype=torch.int32, device='cuda')
    bt = bt.view(B, max_seqlen_pad // pbs)
    bt = bt.view(-1)[torch.randperm(bt.numel(), device='cuda')].view(B, -1)

    bk = torch.randn(bt.numel(), pbs, hkv, d, dtype=torch.bfloat16, device='cuda') / 10
    bk.clamp_(min=-1.0, max=1.0)

    for i in range(B):
        cur_len = int(cache_seqlens_cpu[i].item())
        cnb = kk.cdiv(cur_len, pbs)
        bk[bt[i][cnb:]] = float('nan')
        if cur_len % pbs != 0:
            bk[bt[i][cnb - 1]][cur_len % pbs:] = float('nan')
        bt[i][cnb:] = 2147480000

    cache_seqlens = cache_seqlens_cpu.cuda()
    return cache_seqlens, q, bt, bk


# ═══════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════

def run_one_test(cfg: dict) -> Tuple[bool, str]:
    B, sq, sk, hq, hkv = cfg['B'], cfg['sq'], cfg['sk'], cfg['hq'], cfg['hkv']
    causal, varlen, zero_len = cfg['causal'], cfg.get('varlen', False), cfg.get('zero_len', False)
    d, dv, pbs = 576, 512, 64
    seed = cfg.get('seed', 42)

    if hq % hkv != 0:
        return True, "SKIP"

    cache_seqlens, q, bt, bk = generate_test_data(
        B, sq, sk, hq, hkv, d, dv, pbs, varlen, zero_len, seed
    )

    # ═══ SM86 kernel on RTX 3090 ═══
    md, _ = flash_mla.get_mla_metadata()
    o_kernel, lse_kernel = flash_mla.flash_mla_with_kvcache(
        q, bk, bt, cache_seqlens, dv, md, None, causal=causal
    )

    # ═══ Hopper reference (pure PyTorch, runs on any device) ═══
    o_ref, lse_ref = reference_torch(
        cache_seqlens, bt, q, bk, dv, causal
    )

    # Both on GPU for comparison
    o_k = o_kernel.float()
    o_r = o_ref.float().cuda()
    lse_k = lse_kernel.float()
    lse_r = lse_ref.float().cuda()

    # Check correctness
    ok_out = kk.check_is_allclose(
        'out', o_kernel, o_ref.cuda(),
        abs_tol=8e-4, rel_tol=2.01 / 128, cos_diff_tol=5e-6
    )
    ok_lse = kk.check_is_allclose(
        'lse', lse_kernel, lse_ref.cuda(),
        abs_tol=1e-6, rel_tol=8.01 / 65536
    )
    no_nan = not (torch.isnan(o_kernel).any() or torch.isnan(lse_kernel).any())

    if ok_out and ok_lse and no_nan:
        return True, "PASS"
    msgs = []
    if not ok_out:
        diff = (o_k - o_r).abs()
        msgs.append(f"OUT max_diff={diff.max():.6f}")
    if not ok_lse:
        diff = (lse_k - lse_r).abs()
        msgs.append(f"LSE max_diff={diff.max():.6f}")
    if not no_nan:
        msgs.append("NaN")
    return False, " ".join(msgs)


# ═══════════════════════════════════════════════════════════════
# Test case grid
# ═══════════════════════════════════════════════════════════════

def build_cases():
    cases = []

    # Edge cases first
    edges = [
        {'B': 1, 'sq': 1, 'sk': 1,   'hq': 1,   'hkv': 1, 'causal': False, 'tag': 'edge_min'},
        {'B': 1, 'sq': 1, 'sk': 64,  'hq': 128, 'hkv': 1, 'causal': False, 'zero_len': True, 'tag': 'edge_zerolen'},
        {'B': 1, 'sq': 1, 'sk': 0,   'hq': 128, 'hkv': 1, 'causal': False, 'zero_len': True, 'tag': 'edge_allzero'},
        {'B': 1, 'sq': 1, 'sk': 65,  'hq': 128, 'hkv': 1, 'causal': False, 'tag': 'edge_partial_block'},
        {'B': 1, 'sq': 1, 'sk': 127, 'hq': 128, 'hkv': 1, 'causal': False, 'tag': 'edge_partial_block2'},
        {'B': 1, 'sq': 1, 'sk': 64,  'hq': 128, 'hkv': 1, 'causal': False, 'tag': 'edge_exact_one_block'},
        {'B': 1, 'sq': 1, 'sk': 128, 'hq': 128, 'hkv': 1, 'causal': False, 'tag': 'edge_two_blocks'},
        {'B': 1, 'sq': 1, 'sk': 128, 'hq': 9,   'hkv': 1, 'causal': False, 'tag': 'edge_partial_mblock'},
        {'B': 1, 'sq': 1, 'sk': 128, 'hq': 63,  'hkv': 1, 'causal': False, 'tag': 'edge_partial_mblock2'},
        {'B': 1, 'sq': 1, 'sk': 64,  'hq': 128, 'hkv': 2, 'causal': False, 'tag': 'edge_gqa2'},
        {'B': 1, 'sq': 1, 'sk': 64,  'hq': 128, 'hkv': 8, 'causal': False, 'tag': 'edge_gqa8'},
        {'B': 1, 'sq': 1, 'sk': 64,  'hq': 64,  'hkv': 1, 'causal': False, 'tag': 'edge_hq64'},
        {'B': 4, 'sq': 1, 'sk': 128, 'hq': 128, 'hkv': 1, 'causal': False, 'varlen': True, 'tag': 'edge_varlen'},
        {'B': 128, 'sq': 1, 'sk': 4096, 'hq': 128, 'hkv': 1, 'causal': False, 'tag': 'edge_large_batch'},
        {'B': 1, 'sq': 1, 'sk': 8192, 'hq': 128, 'hkv': 1, 'causal': False, 'tag': 'edge_long_seq'},
    ]
    cases.extend(edges)

    # Combinatorial grid (subset for reasonable runtime)
    for B in [1, 2, 4]:
        for sq in [1]:
            for sk in [64, 128, 512]:
                for hq in [64, 128]:
                    for hkv in [1]:
                        for causal in [False, True]:
                            if len(cases) > 200:
                                break
                            cases.append({'B': B, 'sq': sq, 'sk': sk, 'hq': hq, 'hkv': hkv,
                                          'causal': causal, 'tag': 'basic'})

    # MTP cases (sq=2 with causal)
    for sq in [2]:
        for sk in [128, 256]:
            for hq in [128]:
                for hkv in [1]:
                    for causal in [False, True]:
                        cases.append({'B': 1, 'sq': sq, 'sk': sk, 'hq': hq, 'hkv': hkv,
                                      'causal': causal, 'tag': 'mtp'})

    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    prop = torch.cuda.get_device_properties(args.device)
    print(f'GPU: {prop.name}  |  SM {prop.major}.{prop.minor}  |  {prop.total_memory // 1024**2} MiB')
    print(f'Torch {torch.__version__}  CUDA {torch.version.cuda}')
    print(f'SM86 kernel  ─vs─  Hopper reference (pure PyTorch)')
    print()

    all_cases = build_cases()
    cases = [c for c in all_cases if args.quick and c['tag'].startswith('edge_')] if args.quick else all_cases
    print(f'Running {len(cases)} test cases...')
    print()

    passed, failed, skipped = 0, 0, 0
    for i, cfg in enumerate(cases):
        ok, msg = run_one_test(cfg)
        if msg == 'SKIP':
            skipped += 1
            continue
        if ok:
            passed += 1
            if args.verbose:
                print(f'[{i+1:3d}/{len(cases)}] {cfg["tag"]:25s}  B={cfg["B"]:3d} sq={cfg["sq"]} sk={cfg["sk"]:5d} hq={cfg["hq"]:3d} hkv={cfg["hkv"]} causal={cfg["causal"]}  {msg}')
        else:
            failed += 1
            print(f'[{i+1:3d}/{len(cases)}] {cfg["tag"]:25s}  B={cfg["B"]:3d} sq={cfg["sq"]} sk={cfg["sk"]:5d} hq={cfg["hq"]:3d} hkv={cfg["hkv"]} causal={cfg["causal"]}  {msg}')

    print()
    print(f'═══ Results: {passed} passed, {failed} failed, {skipped} skipped ═══')
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
