# 通用 SIMD / SIMT CostModel 合同

## 1. 适用范围

模型以变换后的 TTIR stage、资源工作量、运行时调度规模和硬件微基准为输入。在线计算禁止读取算子名、kernel 名、源码路径、完整 shape 签名，也禁止使用某个 kernel 的整核截距或误差倍率。

`solve_tril` 属于 `triangular_recurrence` 留出族，只用于验收。`softmax`、scan、indirect gather、FZE/dot recurrence 也属于留出族。参数开发只使用 elementwise、memory microbenchmark 和 synthetic reduction 三个独立算子族。

## 2. 单次迭代的 issue 下界

模式 `m` 的发射指令数按每个 TTIR operation 分别计数，不能先把不同 operation 的元素数相加：

```text
N_issue^m = N_scalar
          + sum_op occurrence(op) * ceil(elements(op) / W_m)
          + sum_pred occurrence(pred) * ceil(elements(pred) / W_m)
          + N_memory^m + N_shared/private + N_control

C_issue^m(1 iter) = N_issue^m / R_issue^m
```

当前硬件参数为：

```text
SIMD: W_SIMD = 64, R_issue_SIMD = 3.298629007 instruction/tick
SIMT: W_SIMT = 32, R_issue_SIMT = 3.298581191 instruction/tick
```

SIMT 的一条 issue 指令是覆盖一个 32-lane warp 的指令。`R_issue_SIMT` 是该执行路径前端的 warp-instruction 发射率，不是“单个 warp 的功率”。SIMD 与 SIMT 的 `R_issue` 数值接近，只能推出同一种逐元素操作的 issue-only 下界中，一条 SIMD 指令最多覆盖两倍元素；不能推出整段 SIMD 一定比 SIMT 快两倍，因为 load/store、依赖链、reduction、dot 和 occupancy 可能成为主瓶颈。

## 3. 单次迭代的执行资源

两种模式分别使用自己的硬件吞吐和启动参数：

```text
C_compute^m = sum_op occurrence(op)
              * instruction_count^m(op)
              / throughput^m(op)

C_exec,serial^m = C_scalar + C_load + C_store + C_shared
                + C_compute + C_predicate + C_reduction
                + C_shuffle + C_dot
```

`instruction_count^SIMD(op)` 通常为 `ceil(elements/W_SIMD)`；`instruction_count^SIMT(op)` 通常为 lane/warp lowering 对应的指令数。内存、reduction、dot 和布局转换优先使用可复用的硬件原语微基准；未命中时回退到按字节、warp instruction 或 FLOP 的解析公式。

## 4. Stage 公式

设 `k_stage` 为 stage 的语义迭代次数。`C_exec(1 iter)` 和 `C_issue(1 iter)` 均不含这个循环次数，stage 组合时统一乘一次。

普通串行 stage：

```text
C_stage^m = C_setup^m
          + k_stage * max(C_exec,serial^m(1 iter), C_issue^m(1 iter))
```

有 lowering/schedule 证据的 SIMD independent pipeline：

```text
C_stage^SIMD = C_setup^SIMD
             + k_stage * max(C_load,
                             C_store,
                             C_compute + C_dot + C_reduction + C_shuffle,
                             C_scalar + C_predicate,
                             C_issue)
```

SIMT loop-carried recurrence 的依赖下界：

```text
C_dep^SIMT = max(C_critical + C_load + C_store, C_issue)
C_stage^SIMT,F1 = C_setup + k_stage * max(C_dep, C_group_recurrence)
```

SIMT SuperBlock factor `F` 把 `F` 个完整逻辑程序放入同一物理程序。吞吐率已经按 `base_warps * F` 的 active warps 选择，但 `F` 个程序的指令和字节不能消失：

```text
C_group^SIMT,F = max(C_dep,
                     F * C_exec,serial,
                     F * C_issue,
                     C_group_recurrence)

C_stage^SIMT,F = C_setup
               + k_stage * (C_group^SIMT,F + F * C_uncovered_private_stack)
```

`C_uncovered_private_stack` 只允许来自 lowering 明确给出的 LDK/STK 数量，或通过跨算子留出门禁的通用结构公式。缺失数据时保持为 0，不能使用 solve_tril shape 表或单 kernel 截距补齐。

## 5. 整核时间

route 先按运行时逻辑程序数、物理 core 数、SuperBlock factor 和 tail 计算每个 stage 的执行次数，再组合各 stage 与合法 handoff：

```text
C_route = compose(runtime-scaled stage costs, proven overlap, handoff)
C_kernel_us = (C_route + B(P_physical)) / SYS_CNT_frequency
```

`B(P_physical)` 是空载/边界微基准得到的、只依赖物理程序数的共享 kernel 头尾开销曲线。它对 SIMD、SIMT 和 Mixed 使用同一条曲线，不参与 route 排序，也不能含 kernel identity。

## 6. 推广门禁

每个 case 运行 auto、SIMD、SIMT F1/F2/F4；支持 Mixed 的 case 另测 Mixed。SIMD 与 SIMT 分别做 operator-family leave-one-family-out，并在从未参与拟合的五个算子族上验收。任一训练族交叉验证、最终训练样本或留出样本的绝对相对误差超过 10%，候选公式状态必须保持 `research_not_promoted`。

离线残差拟合只用于定位缺失资源项，不会自动修改生产 profile。生产公式变更还必须有对应 TTIR/lowering 结构证据、A5 实测、完整 costmodel 回归和 route/correctness/performance 门禁。

## 7. 统一回归矩阵与手工执行

`generalization_matrix.json` 是默认、唯一的完整回归入口。当前矩阵包含 100 个不重名 case：历史主矩阵 71 个、第二批 pytest_ut 结构 holdout 17 个、首批 pytest_ut 结构 holdout 12 个。版本化快照为 `generalization_matrix_v140_all_100.json`；两个 holdout 子集分别保留在 `pytest_ut_structural_17_manifest.json` 和 `pytest_ut_structural_first_12_manifest.json`，便于针对性复测，但它们同时也是主矩阵成员。

在 A5 上先进入目录并加载固定环境：

```bash
cd /data/kaixin/triton-ascend/third_party/ascend/unittest/costmodel_ut/costmodel_generalization
source /data/kaixin/set_env.sh
export TRITON_DEBUG=0
export TRITON_DISABLE_LINE_INFO=1
export ASCEND_RT_VISIBLE_DEVICES=<空闲卡号>
```

列出全部 case，不需要 NPU：

```bash
/home/pkg/ascend_py311/bin/python run_generalization_matrix.py --list-cases
```

按精确名称手工运行一个或多个 case：

```bash
/home/pkg/ascend_py311/bin/python run_generalization_matrix.py \
  --case pytest_add_f32_2x4096x8 \
  --output-root /data/kaixin/manual_costmodel/pytest_add

/home/pkg/ascend_py311/bin/python run_generalization_matrix.py \
  --case pytest_add_f32_2x4096x8 \
  --case pytest_chunk_cumsum_f32_b4_t2048_h8_bt128 \
  --output-root /data/kaixin/manual_costmodel/two_cases
```

手工运行完整 100-case Auto 回归：

```bash
/home/pkg/ascend_py311/bin/python run_generalization_matrix.py \
  --output-root /data/kaixin/manual_costmodel/all_100_auto
```

`--mode simd`、`--mode simt_only:1`、`--mode simt_only:2`、`--mode simt_only:4` 可以指定路线；`--only-family FAMILY` 可以筛选结构族；`--shard-index I --shard-count N` 可以把同一主矩阵拆成多个手工 shard。每次执行都会在输出目录保存 manifest/profile 快照、源码状态、编译器信息、route JSON、正确性、profiler 数据和退出码。
