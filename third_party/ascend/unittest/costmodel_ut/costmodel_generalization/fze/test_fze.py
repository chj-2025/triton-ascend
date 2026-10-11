r"""
Flatten = Zipped * Experts
"""

from typing import Optional

import torch
import triton
import triton.language as tl
import torch_npu
#from triton.language.extra.ascend import libdevice

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64

ALLOW_TF32 = False
ZMOE_DETERMINISTIC = False


def FZE_TORCH(
    A,
    A_eid,
    A_idx,
    A_blk_idx,
    A_weight=None,
    B=None,
    C=None,
):
    r"""
    Input:
    A (Zipped data): [ BS, H ]
    A_eid (Zipped expert indices for the kernel): [ M ] int
    A_idx (Zipped index): [ M ] int
    A_blk_idx (Zipped block indicator): [ B ] int
    A_weight (Not used in this configuration): None
    B (Experts weights): [ E, H, H2 ]
    C (Specified output, optional): [ M, H2 ]

    Output:
    C (same as above)
    """
    # 检查输入维度
    BS, H = A.size()
    M = A_eid.size(0)
    B_blk = A_blk_idx.size(0)

    assert A_idx.size(0) == M, "A_idx shape mismatch with A_eid"
    assert B is not None, "B (experts weights) must be provided"
    E, H_B, H2 = B.size()
    assert H == H_B, f"A's hidden size {H} doesn't match B's hidden size {H_B}"

    # 检查A_weight是否为None
    if A_weight is not None:
        print("Warning: A_weight is provided but not used in this configuration")

    # 处理输出张量
    if C is None:
        C = torch.empty((M, H2), device=A.device, dtype=A.dtype)
    else:
        assert C.size(0) == M and C.size(1) == H2, "Output shape mismatch"

    # 处理块的起始和结束索引
    # 确保A_blk_idx是递增的
    sorted_indices = torch.argsort(A_blk_idx)
    sorted_block_starts = A_blk_idx[sorted_indices]

    # 添加最后一个块的结束索引
    block_boundaries = torch.cat([sorted_block_starts, torch.tensor([M], device=A.device)])

    # 处理每个块
    for i in range(B_blk):
        start = block_boundaries[i]
        end = block_boundaries[i + 1]

        # 跳过空块
        if start >= end:
            continue

        # 获取当前块的所有元素
        block_elements = torch.arange(start, end, device=A.device)

        # 获取当前块的专家索引和数据索引
        block_eids = A_eid[block_elements]
        block_indices = A_idx[block_elements]

        # 对每个专家分别处理
        for e in torch.unique(block_eids):
            # 找到使用当前专家的元素
            expert_mask = (block_eids == e)
            if not expert_mask.any():
                continue

            # 获取这些元素的索引
            expert_elements = block_elements[expert_mask]
            expert_data_indices = block_indices[expert_mask]

            # 提取输入数据
            input_data = A[expert_data_indices]

            # 应用专家权重矩阵
            expert_output = torch.matmul(input_data, B[e])

            # 存储结果
            C[expert_elements] = expert_output

    return C


def FZE(
    A,
    A_eid,
    A_idx,
    A_blk_idx,
    A_weight,
    B,
    C=None,
):
    r"""
    Input:
    A (Zipped data): [ BS, H ]
    A_eid (Zipped expert indices for the kernel): [ Permuted ] int
    A_idx (Zipped index): [ Permuted ] int
    A_blk_idx (Zipped block indicator): [ Permuted ] int
    A_weight (Zipped weights, optional): [ Permuted ]
    B (Experts): [ E, H, H2 ]
    C (Specified output, optional): [ Permuted, H2 ]

    Output:
    C (same as above)
    """
    if C is None:
        C = torch.empty((A_idx.size(0), B.size(-1)), device=A.device, dtype=A.dtype)
    else:
        assert C.size(0) == A_idx.size(0) and C.size(1) == B.size(-1)

    FZE_compileable(C, A, A_eid, A_idx, A_blk_idx, A_weight, B)
    return C


def FZE_compileable(
    O: torch.Tensor,
    A: torch.Tensor,
    A_eid: torch.Tensor,
    A_idx: torch.Tensor,
    A_blk_idx: torch.Tensor,
    A_weight: Optional[torch.Tensor],
    B: torch.Tensor,
) -> None:

    def grid(META):
        grid_num = (
            triton.cdiv(B.size(-1), BLOCK_N),
            A_blk_idx.size(0),
        )
        return grid_num

    _FZE[triton.cdiv(B.size(-1), BLOCK_N), A_blk_idx.size(0)](
        # A_ptr, stride_am, stride_ak,
        A,
        A.stride(0),
        A.stride(1),
        # A_eid_ptr, A_idx_ptr, A_blk_idx_ptr, A_weight_ptr,
        A_eid,
        A_idx,
        A_blk_idx,
        A_weight,
        A_weight is not None,
        # B_ptr, stride_be, stride_bk, stride_bn,
        B,
        B.stride(0),
        B.stride(1),
        B.stride(2),
        # O_ptr, stride_om, stride_on,
        O,
        O.stride(0),
        O.stride(1),
        M=A_idx.size(0),
        K=A.size(1),
        N=B.size(-1),
        E=B.size(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        ACC_TYPE=tl.float32,
        allow_tf32=ALLOW_TF32,
        NO_K_MASK=True,
        NO_N_MASK=True,
    )


def _config_fze():
    config = [triton.Config({'BLOCK_N': BN, 'BLOCK_K': BK}) for BN in [64] for BK in [64]]

    if ZMOE_DETERMINISTIC:
        return config[-1:]
    else:
        return config


@triton.jit
def _FZE(
    A_ptr,
    stride_am,
    stride_ak,
    A_eid_ptr,
    A_idx_ptr,
    A_blk_idx_ptr,
    A_weight_ptr,
    has_w: tl.constexpr,
    B_ptr,
    stride_be,
    stride_bk,
    stride_bn,
    O_ptr,
    stride_om,
    stride_on,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    allow_tf32: tl.constexpr,
    NO_K_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
):
    N_block_id = tl.program_id(axis=0).to(tl.int64)
    M_range = tl.arange(0, BLOCK_M)
    M_block_id = tl.program_id(axis=1).to(tl.int64)
    block_start_idx = tl.load(A_blk_idx_ptr + M_block_id)
    if block_start_idx >= M:
        return
    M_block = tl.max_contiguous(block_start_idx + M_range, BLOCK_M)
    E_idxs = tl.load(A_eid_ptr + M_block, mask=M_block < M, other=E)
    E_idx = tl.min(E_idxs)
    E_mask = E_idxs == E_idx
    A_idx = tl.load(A_idx_ptr + M_block, mask=E_mask, other=0)
    K_block = tl.arange(0, BLOCK_K)
    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N

    A_blk_ptrs = A_ptr + A_idx[:, None] * stride_am + K_block[None, :] * stride_ak
    B_blk_ptrs = B_ptr + K_block[:, None] * stride_bk + N_block[None, :] * stride_bn + E_idx * stride_be

    if has_w:
        A_weight = tl.load(A_weight_ptr + M_block, mask=E_mask, other=0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    iters = tl.cdiv(K, BLOCK_K)
    for K_block_id in range(0, iters):
        if NO_K_MASK:
            a = tl.load(A_blk_ptrs, mask=E_mask[:, None])
            b = tl.load(B_blk_ptrs, mask=N_mask[None, :])
        else:
            K_mask = (K_block_id * BLOCK_K + K_block) < K
            a = tl.load(A_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :])
            b = tl.load(B_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])
        if has_w:
            a = a * A_weight[:, None]
        A_blk_ptrs += BLOCK_K * stride_ak
        B_blk_ptrs += BLOCK_K * stride_bk
        # acc += tl.dot(a, b, allow_tf32=allow_tf32, out_dtype=ACC_TYPE)
        acc += tl.dot(a, b, out_dtype=ACC_TYPE)

    O_blk_ptrs = O_ptr + (M_block[:, None] * stride_om + N_block[None, :] * stride_on)
    acc = acc.to(O_blk_ptrs.dtype.element_ty)
    tl.store(O_blk_ptrs, acc, mask=E_mask[:, None] & N_mask[None, :])


def get_device():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.device('npu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


if __name__ == '__main__':
    '''
    Input:
    A (Zipped data): [ BS, H ]
    A_eid (Zipped expert indices for the kernel): [ Permuted ] int
    A_idx (Zipped index): [ Permuted ] int
    A_blk_idx (Zipped block indicator): [ Permuted ] int
    A_weight (Zipped weights, optional): [ Permuted ]
    B (Experts): [ E, H, H2 ]
    C (Specified output, optional): [ Permuted, H2 ]

    Output:
    C (same as above)
    '''
    device = get_device()

    # path = "ZFE_input_args_rank0.pth"
    # inputs = torch.load(path, map_location='npu')

    torch.manual_seed(0)
    A = torch.randn((590, 1024), device=device)
    B = torch.randn((16, 1024, 1536), device=device)
    A_eid = torch.tensor([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8,
        10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
        11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
        11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
        11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
        13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
        13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14,
        14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
    ], dtype=torch.int64, device=device)
    A_idx = torch.tensor([
        4, 125, 137, 268, 270, 273, 275, 277, 278, 282, 286, 288, 292, 295, 296, 303, 343, 355, 361, 371, 404, 406, 426,
        429, 498, 533, 16, 19, 30, 65, 103, 139, 153, 271, 432, 449, 457, 458, 468, 479, 480, 482, 484, 486, 489, 512,
        515, 516, 519, 520, 524, 526, 528, 531, 535, 539, 542, 547, 143, 297, 299, 301, 308, 310, 313, 317, 318, 320,
        322, 324, 326, 347, 357, 359, 363, 366, 368, 373, 378, 380, 382, 384, 496, 575, 579, 13, 24, 26, 33, 34, 36, 38,
        44, 47, 48, 50, 57, 61, 62, 68, 72, 74, 85, 86, 98, 107, 108, 110, 118, 120, 122, 134, 142, 146, 148, 160, 172,
        175, 179, 184, 191, 196, 199, 208, 211, 215, 221, 227, 233, 239, 245, 251, 257, 259, 263, 331, 387, 465, 6, 8,
        11, 12, 14, 22, 25, 27, 35, 37, 41, 46, 49, 51, 52, 58, 60, 63, 64, 66, 71, 73, 75, 83, 84, 87, 92, 94, 96, 99,
        101, 105, 106, 109, 111, 113, 116, 121, 123, 124, 129, 130, 133, 135, 136, 140, 144, 147, 150, 154, 157, 159,
        163, 166, 169, 178, 187, 190, 202, 214, 223, 226, 235, 238, 241, 247, 250, 262, 267, 280, 283, 284, 291, 302,
        304, 306, 311, 312, 314, 316, 321, 323, 325, 327, 329, 330, 332, 334, 336, 339, 340, 342, 344, 346, 349, 350,
        353, 356, 358, 360, 362, 364, 367, 369, 370, 372, 374, 377, 383, 385, 386, 389, 391, 395, 407, 417, 433, 453,
        461, 464, 467, 470, 472, 492, 510, 522, 541, 91, 126, 165, 171, 176, 181, 182, 188, 193, 194, 200, 204, 206,
        212, 216, 218, 224, 228, 230, 236, 240, 242, 248, 252, 254, 260, 264, 266, 411, 423, 447, 497, 501, 537, 551,
        555, 565, 577, 581, 17, 20, 23, 28, 32, 39, 40, 43, 45, 53, 55, 76, 78, 82, 88, 90, 100, 112, 114, 117, 119,
        141, 145, 149, 151, 161, 162, 173, 174, 185, 186, 197, 198, 209, 210, 213, 220, 222, 225, 232, 234, 237, 244,
        246, 249, 256, 258, 261, 279, 281, 285, 287, 289, 290, 293, 294, 298, 300, 309, 335, 345, 376, 415, 504, 506,
        583, 502, 505, 0, 2, 7, 10, 15, 56, 80, 93, 104, 155, 269, 272, 274, 276, 307, 315, 319, 328, 333, 337, 338,
        341, 348, 351, 352, 365, 375, 393, 396, 399, 400, 403, 405, 409, 412, 420, 422, 425, 430, 434, 436, 439, 441,
        442, 444, 446, 448, 451, 452, 454, 456, 462, 466, 469, 473, 474, 476, 478, 481, 483, 485, 488, 490, 494, 500,
        503, 513, 514, 521, 525, 527, 530, 532, 534, 536, 538, 540, 543, 545, 546, 548, 550, 552, 554, 556, 558, 560,
        562, 564, 566, 568, 570, 572, 574, 576, 578, 580, 582, 584, 586, 588, 9, 18, 21, 29, 31, 42, 54, 59, 70, 77, 79,
        89, 95, 102, 115, 127, 131, 138, 203, 5, 67, 69, 81, 97, 128, 132, 152, 156, 158, 164, 167, 168, 170, 177, 180,
        183, 189, 192, 195, 201, 205, 207, 217, 219, 229, 231, 243, 253, 255, 265, 354, 379, 381, 388, 390, 392, 394,
        397, 398, 401, 402, 408, 410, 413, 414, 416, 418, 421, 424, 427, 428, 431, 437, 438, 443, 471, 475, 477, 493,
        495, 499, 517, 529, 553, 557, 561, 563, 567, 569, 571, 1, 3, 419, 435, 440, 487, 507, 509, 518, 523, 544, 585,
        587, 589, 305, 445, 450, 455, 459, 460, 463, 491, 508, 511, 549, 559, 573
    ], dtype=torch.int64, device=device)
    A_blk_idx = torch.tensor(
        [0, 26, 58, 85, 138, 202, 265, 293, 304, 368, 369, 370, 372, 436, 473, 492, 556, 563, 577, 0, 0, 0, 0, 0, 0, 0],
        dtype=torch.int64, device=device)
    # A = inputs['arg0_A'].to(device)
    # B = inputs['arg1_B'].to(device)
    # A_eid = inputs['arg2_O_eid'].to(device)
    # A_idx = inputs['arg3_O_idx'].to(device)
    # A_blk_idx = inputs['arg4_O_blk_idx'].to(device)
    A_weight = None
    bs = 590

    # breakpoint()
    print("====A_eid", A_eid)
    print("====A_idx", A_idx)
    print("====A_blk_idx", A_blk_idx)

    C = FZE(
        A,
        A_eid,
        A_idx,
        A_blk_idx,
        A_weight,
        B,
    )

    C_torch = FZE_TORCH(A, A_eid, A_idx, A_blk_idx, None, B)
    # torch.npu.synchronize()
    print("====C ", C.shape, C)
    print("====C_torch ", C_torch.shape, C_torch)

    print("====diff ", (C - C_torch).abs().max())
    print("====acc ", torch.allclose(C, C_torch, atol=1e-2, rtol=1e-2))

    # profiler_wrapper(FZE, A, A_eid, A_idx, A_blk_idx, A_weight, B)
