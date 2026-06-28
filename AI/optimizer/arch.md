# 端侧模型部署优化参考手册

> 一套与具体框架/实现解耦的**通用端侧模型部署优化方法论**。
> 目标：指导任意端侧模型在资源受限 SoC 上的部署优化全流程。
> 主参考硬件：**Jetson Thor（Blackwell / 统一内存）**；次参考：**地平线 征程/旅程（BPU + DSP）**。
> 贯穿 MVP 用例：**π0.5（PaliGemma ViT + Gemma LLM + flow-matching action expert）**。
>
> 本手册只讲"怎么想、按什么顺序做、用什么工具量"，不绑定任何一套代码实现。

---

## 目录

1. [心智模型：优化维度 × 优化层次](#1-心智模型优化维度--优化层次)
2. [端侧硬件抽象与平台画像](#2-端侧硬件抽象与平台画像)
3. [整体系统架构（部署流水线全景）](#3-整体系统架构部署流水线全景)
4. [核心工作流程（profile 驱动的优化闭环）](#4-核心工作流程profile-驱动的优化闭环)
5. [维度一：内存优化](#5-维度一内存优化)
6. [维度二：计算优化](#6-维度二计算优化)
7. [维度三：调度优化](#7-维度三调度优化)
8. [维度四：Profiling 体系](#8-维度四profiling-体系)
9. [两种部署范式：编译器引擎 vs 手写 runtime](#9-两种部署范式编译器引擎-vs-手写-runtime)
10. [硬件架构优化](#10-硬件架构优化)
11. [系统优化](#11-系统优化)
12. [工具矩阵](#12-工具矩阵)
13. [MVP 贯穿案例：π0.5 on Thor](#13-mvp-贯穿案例π05-on-thor)
14. [决策原则与反模式](#14-决策原则与反模式)
15. [多模型并行编排与混跑性能估算](#15-多模型并行编排与混跑性能估算)
    - [15.3.1 深入：弱抢占（"may preempt" 不保证）](#1531-深入弱抢占may-preempt-不保证)
    - [15.3.2 深入：big.LITTLE CPU（隐形瓶颈）](#1532-深入biglittle-cpu隐形瓶颈)

---

## 1. 心智模型：优化维度 × 优化层次

端侧部署优化本质是一个**二维问题**。任何一个具体优化项，都同时落在某个"优化维度"和某个"优化层次"上。先建立这张坐标系，后面所有手段都能对号入座。

### 1.1 四个优化维度（优化"什么"）

| 维度 | 核心问题 | 典型瓶颈信号 | 主要抓手 |
|------|----------|--------------|----------|
| **内存** | 数据放哪、占多少、搬多少次 | OOM、显存峰值、带宽打满、cache miss | 量化、权重布局、KV cache、复用/零拷贝 |
| **计算** | 算力够不够、有没有空算 | Tensor Core 利用率低、SM 空转 | 低精度、算子融合、tiling、kernel 选型 |
| **调度** | kernel 怎么排、host 等不等 | kernel 数过多、launch 占比高、device 空泡 | 整图化、CUDA Graph、流水线、异步 |
| **Profiling** | 瓶颈到底在哪 | （无瓶颈信号 = 没量化） | 计时、roofline、NVTX、trace 分析 |

> **关键洞察**：Profiling 不是"第四类优化"，而是前三类的**前置条件**。没有量化数据的优化都是猜测。

### 1.2 四个优化层次（在"哪一层"动手）

| 层次 | 改什么 | 收益来源 | 风险/成本 |
|------|--------|----------|-----------|
| **编译优化** | 计算图、算子、精度、内核选型 | 图融合、tactic 搜索、量化 | 中；依赖编译器能力 |
| **硬件架构优化** | 贴合具体 SoC 的内存层级/计算单元 | 用满 Tensor Core/TMA/统一内存 | 高；强平台绑定 |
| **系统优化** | OS、内存分配、电源、并发 | 去掉系统级开销与抖动 | 低-中；易被忽视 |
| **工具** | 度量与定位手段本身 | 加速迭代、避免盲目优化 | 一次性投入 |

### 1.3 二维矩阵（手段定位表）

> 横轴=层次，纵轴=维度。每格给典型手段，后续章节展开。

| | 编译优化 | 硬件架构优化 | 系统优化 |
|---|---|---|---|
| **内存** | 量化(FP8/NVFP4/INT4)、权重布局重排、常量折叠 | 统一内存零拷贝、HBM/LPDDR 带宽对齐、片上 SRAM 复用 | pinned memory、大页(hugepage)、allocator 复用 |
| **计算** | 算子融合、低精度 GEMM、tiling/autotune | Tensor Core/TMA/Warp Specialization、专用单元(BPU/DSP/NPU) | 频率锁定(jetson_clocks)、功耗模式(nvpmodel) |
| **调度** | 整图编译、kernel 数压缩、流水线融合 | CUDA Graph、PDL 链式启动、多 stream | 进程亲和性、IRQ/CPU 隔离、实时调度 |
| **Profiling** | 图 dump、layer-wise timing | NVTX、硬件计数器(ncu)、SOL% | nsys 系统级 trace、功耗/温度遥测 |

---

## 2. 端侧硬件抽象与平台画像

端侧优化高度依赖目标硬件。先把硬件抽象成几个"决定优化策略的关键参数"，再看具体平台。

### 2.1 决定优化策略的硬件参数

| 参数 | 为什么决定优化方向 |
|------|---------------------|
| **算力 (TOPS/TFLOPS) × 精度档** | 决定低精度量化的天花板（支持 FP8/FP4？） |
| **内存带宽 (GB/s)** | decode/访存型负载的硬上限；`每 token 时间 ≈ 权重大小 / 带宽` |
| **内存模型（离散显存 vs 统一内存）** | 决定是否有 PCIe 墙、是否能零拷贝 |
| **片上 SRAM / L2 容量** | 决定 tiling 策略与融合粒度 |
| **计算单元类型** | 通用 GPU vs 专用 NPU/BPU/DSP → 完全不同的工具链 |
| **kernel launch 开销** | 决定小算子是否要整图化/CUDA Graph |
| **功耗/散热预算** | 端侧持续算力 << 峰值算力，需锁频标定 |

> **第一性原理**：先用 [SysPeek](#12-工具矩阵) 之类工具测出目标平台**实际可达**的算力/带宽/launch 开销，而不是用 datasheet 峰值。纸面峰值与实测常差 2-3×。

### 2.2 平台画像对比

| | **Jetson Thor（主参考）** | **地平线 征程/旅程** | 桌面独显（开发对照） |
|---|---|---|---|
| 计算单元 | Blackwell GPU + Tensor Core | **BPU**（专用 NN 加速器）+ DSP + ARM | 离散 GPU |
| 内存模型 | **统一内存**（CPU/GPU 共享 LPDDR） | 统一内存（BPU/CPU 共享 DDR） | 离散（PCIe 隔离主存/显存） |
| 内存带宽 | LPDDR ~273 GB/s | DDR，带宽更受限 | GDDR/HBM，几百 GB/s~TB/s |
| 低精度 | FP8 / **NVFP4(W4A4)** / INT8 | **INT8 为主**，部分 INT16/FP16 | FP8(SM89+)/FP4(SM100+) |
| 工具链 | CUDA / TensorRT / nsys / ncu | **天工开物**（horizon_nn / hbdk 编译器 + 量化工具） | CUDA 全家桶 |
| 优化范式 | 通用图编译 + 可手写 CUDA | **量化优先 + 算子受限**（必须落到 BPU 支持算子） | 同 Thor |
| 部署形态 | TRT engine / 手写 runtime | hbm 模型 + 板端 runtime | — |

**两类端侧 SoC 的根本差异**：

- **GPU 类（Thor）**：算子表达自由，优化重点在"低精度 + 融合 + 调度"；可以手写 kernel 突破编译器天花板。
- **NPU/BPU 类（地平线）**：算子集**受限且固定**，优化重点前移到"**模型结构必须 BPU 友好 + 量化必须过编译器**"。不被 BPU 支持的算子会回退到 CPU/DSP，成为性能悬崖。**结构设计期就要对齐算子白名单**。

> 通用原则：**离硬件越近、算子越受限的平台，优化越要往前端（模型结构/量化）移**。Thor 上能事后用 kernel 救，BPU 上往往只能改图重训。

---

## 3. 整体系统架构（部署流水线全景）

把"从训练产物到端侧实时推理"抽象成一条与实现无关的流水线。任何框架（TensorRT、地平线天工开物、TVM、ONNX Runtime、自研 runtime）都能套进这个骨架。

```text
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 A：模型准备（离线，host/dev 机）                                  │
│  训练产物(HF/PyTorch ckpt)                                            │
│     │                                                                 │
│     ├─[A1] 结构适配：算子白名单对齐、动态轴规整、控制流消除             │
│     ├─[A2] 校准集构建：分层采样、覆盖真实分布（决定量化精度上限）       │
│     ├─[A3] PTQ 量化：FP8 / NVFP4 / INT8 / W4A8 / KV-cache 量化         │
│     └─[A4] 图导出：ONNX / 中间表示 + sidecar(scale/词表/prompt)        │
└─────────────────────────────────────────────────────────────────────┘
                            │  中间表示 (IR/ONNX) + 量化元数据
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 B：编译构建（离线，目标架构）                                     │
│     ├─[B1] 图优化：算子融合、常量折叠、layout 选择、死代码消除          │
│     ├─[B2] 内核选型：tactic autotune / 专用 kernel / 预生成 artifact   │
│     ├─[B3] 精度落地：插入 Q/DQ、epilogue descale、混精策略             │
│     └─[B4] 产出引擎：TRT engine / hbm / 手写 pipeline 权重 repack      │
│     ⚠ 必须为【目标 SoC 架构】编译（如 Thor sm_110），勿 fallback        │
└─────────────────────────────────────────────────────────────────────┘
                            │  可执行引擎 + 权重
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 C：运行时执行（在线，板端）                                       │
│     ├─[C1] 内存规划：权重常驻、buffer 预分配、KV cache、零拷贝         │
│     ├─[C2] 调度：CUDA Graph capture/replay、多 stream、流水线          │
│     ├─[C3] 异构编排：各阶段选最优后端（视觉/LLM/动作头分别处理）       │
│     └─[C4] 系统配置：锁频、功耗模式、CPU 亲和、大页                     │
└─────────────────────────────────────────────────────────────────────┘
                            │  trace / 计数器 / 功耗
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 D：度量与回归（贯穿全程）                                         │
│     profiling(nsys/ncu/NVTX) → 瓶颈定位 → 回到 A/B/C 对应阶段          │
│     精度回归(端到端任务指标) + 性能回归(延迟/吞吐/功耗)                 │
└─────────────────────────────────────────────────────────────────────┘
```

**贯穿性原则**：

- **优化项的"开关位"在它最早能生效的阶段**。运行时想要的能力（如 FP8 KV、零拷贝绑定），其元数据往往必须在阶段 A/B 就写对，否则运行时无从生效。
- **阶段 D 不是末尾，是回路**。每一轮优化都从 D 出发（定位）、回到 D 收口（验证）。

---

## 4. 核心工作流程（profile 驱动的优化闭环）

```text
        ┌──────────────────────────────────────────┐
        │ 0. 建 baseline + 平台标定                  │
        │   - 跑通 FP16/BF16 端到端，记延迟/精度      │
        │   - SysPeek 测平台真实算力/带宽/launch     │
        └──────────────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────────────┐
        │ 1. Profile：定位瓶颈                        │◄─────────┐
        │   - nsys 看时间线：哪段最久、device 空不空  │          │
        │   - 分类：compute / memory / launch / sync  │          │
        └──────────────────────────────────────────┘          │
                          │                                     │
                          ▼                                     │
        ┌──────────────────────────────────────────┐          │
        │ 2. 归因：瓶颈属于哪个维度？                  │          │
        │   roofline：在算力屋顶还是带宽屋顶？         │          │
        │   kernel 数 vs 实算时间：launch-bound？      │          │
        └──────────────────────────────────────────┘          │
                          │                                     │
                          ▼                                     │
        ┌──────────────────────────────────────────┐          │
        │ 3. 选手段：按 ROI 排序（见 §14 决策原则）   │          │
        │   先做"低风险高收益"、再"高风险"            │          │
        └──────────────────────────────────────────┘          │
                          │                                     │
                          ▼                                     │
        ┌──────────────────────────────────────────┐          │
        │ 4. 实施 + 验证                              │          │
        │   - 性能：延迟/吞吐/功耗有没有降            │          │
        │   - 精度：端到端任务指标没掉（先精度后性能） │──────────┘
        └──────────────────────────────────────────┘   迭代直到达标
```

### 4.1 瓶颈四分类（归因的核心）

| 分类 | 判据 | 对应维度 | 典型解法 |
|------|------|----------|----------|
| **compute-bound** | Tensor Core 利用率高、算术强度高（大 GEMM、prefill） | 计算 | 低精度、用满 Tensor Core、更优 tiling |
| **memory-bound** | 带宽打满、算术强度低（decode、逐 token、elementwise） | 内存 | 量化降带宽、融合减少 DRAM 往返、KV 压缩 |
| **launch-bound** | kernel 数 ≫ 实算时间、device 有空泡、host 占比高 | 调度 | 整图化、CUDA Graph、kernel 融合减少数量 |
| **sync-bound** | 频繁 device↔host 同步、`.item()`/`.cpu()`、动态 shape | 调度 | 去同步、静态化 shape、异步流水 |

> **次序铁律**：先归因再下手。memory-bound 的负载上 NVFP4 收益有限；launch-bound 的负载光量化没用、得砍 kernel 数。**用错维度的优化等于做无用功甚至 regress。**

### 4.2 精度门控（性能优化的护栏）

每一轮性能优化都必须过精度回归，且**精度先行**：

1. 先用更稳的校准（分层采样、percentile、真实数据 recalibration）把量化精度做到位。
2. 再做激进的低精度/融合，每步对照端到端任务指标（不是只看 cosine 相似度）。
3. 守住一条可回滚的"已知良好"配置。

> 端侧常见陷阱：单看张量 cosine 0.99+ 以为没问题，端到端任务成功率却掉了一截。**必须用任务级指标做 gate。**

---

## 5. 维度一：内存优化

端侧内存是最稀缺资源。内存优化分三个子问题：**占多少**（容量）、**搬多少**（带宽）、**怎么放**（布局/层级）。

### 5.1 内存四类来源（先分账再优化）

| 来源 | 占用特征 | 优化手段 |
|------|----------|----------|
| **权重** | 常驻、最大头 | 量化（FP8 减半、NVFP4/INT4 减至 1/4）、词表裁剪、FP8 embedding |
| **激活值** | 随 batch/序列长度，瞬时 | buffer 预分配复用、原地计算、融合减少中间张量 |
| **KV cache** | 随上下文增长，长 prefix 时巨大 | FP8 KV cache、零拷贝绑定、prompt cache 复用 |
| **引擎/运行时开销** | context scratch、临时副本 | 共享 context memory、double-buffer、削减加载期峰值 |

### 5.2 容量优化：量化（最大杠杆）

量化既降容量又降带宽，是端侧第一杠杆。各档位权衡：

| 精度 | 位宽 | 容量 | 适用 | 风险 |
|------|------|------|------|------|
| FP16/BF16 | 16 | baseline | 精度安全垫 | 无压缩 |
| FP8 (E4M3) | 8 | 1/2 | 通用主力，权重+激活 | per-tensor 需校准 |
| INT8 | 8 | 1/2 | NPU/BPU 主力 | 分布敏感，需非对称/per-channel |
| **NVFP4 (W4A4)** | 4 | 1/4 | Blackwell/Thor、显存紧张 | 位少，靠 block-16 细 scale + AWQ 兜底 |
| W4A8 | 4w/8a | ~1/4 | 算力换内存最佳点 | 需 weight repack |

> 通用要点：
> - **对称量化（zero_point=0）** 适合权重与 Norm 后近零对称的激活；INT8 偏态分布才用非对称。
> - **scale 粒度**：per-tensor 最省但易被 outlier 拉伸；per-channel/block 用细粒度换精度。
> - **细粒度 scale 的本质**：让每个小块都用满低位宽的表示范围，outlier 只影响所在块，避免小值被量化成 0。

### 5.3 带宽优化：减少 DRAM 往返

memory-bound 负载的核心是"**少跑几趟 global memory**"：

- **算子融合**：把 Norm/激活/量化/elementwise 链合并，中间结果留在片上 SRAM/寄存器，不落 DRAM。
- **KV cache 量化**：decode 阶段每步都要读全量 KV，FP8 KV 直接把这条带宽减半。
- **权重布局重排（离线）**：预先转置/交错/合并成 kernel 友好布局，运行时零开销、访存连续。
- **零拷贝/直接绑定**：消除阶段间的中间拷贝（如引擎输出直接写进下游 cache 区）。

> **内存层级直觉**（适用所有 GPU 类平台）：
> ```
> global memory (LPDDR/HBM, 带宽瓶颈所在) → L2 → L1/SMEM → 寄存器/TMEM → 计算单元
> ```
> 权重太大常驻 global memory，每次计算都要 load；片上各级 memory 带宽极高、不是瓶颈，作用是"**少跑 global memory**"。优化访存=提高片上复用率。

### 5.4 统一内存的认知陷阱（Thor/地平线通用）

统一内存（CPU/GPU/BPU 共享一块物理 DRAM）**消掉了 PCIe 墙，但没消掉软件层的 host/device 双地址空间与显式拷贝**：

- 物理 DRAM 只有一份 → host↔device 不过 PCIe；
- 但框架里仍有 host 指针与 device 指针两套分配，`.to(device)` 仍触发逻辑拷贝 + cache 一致性开销；
- **真零拷贝**（mapped pinned / managed memory）存在但编程复杂、有 coherency 成本，默认路径并不走。

> 实测量级（Thor）：device 内部 copy ~255 GB/s（接近 LPDDR 273 峰值），而 host→device 显式 copy 仅 ~127 GB/s。**统一内存 ≠ 自动零拷贝**，要主动设计才能吃到。

### 5.5 系统级内存手段

- **pinned memory**：页锁定主存，GPU 可直接 DMA，避免 staging buffer。上传权重/喂输入用 pinned。
- **大页 (hugepage)**：减少 TLB miss，DriveOS/部分嵌入式平台收益明显。
- **allocator 复用**：缓存式分配器避免运行时 malloc/free churn；热路径**零 malloc**（buffer 预分配一次、整个 session 复用）。
- **加载期峰值削减**：逐层处理后立即释放原权重，避免"原权重 + 量化产物 + 引擎"三者瞬时共存导致 OOM。

---

## 6. 维度二：计算优化

计算优化的目标是"**让计算单元少空转、用满最快的精度档**"。

### 6.1 低精度计算（算力翻倍）

低精度不仅省内存，更直接提升算力吞吐（Tensor Core 在 FP8/FP4 下吞吐成倍于 FP16）。

```
Y = X @ W ≈ (s_x·X_q) @ (s_w·W_q) = (s_x·s_w)·(X_q @ W_q)
            标量 descale            ↑ 低精度 Tensor Core GEMM（FP32 累加）
```

- **descale 是 GEMM 的 epilogue**，不是单独 kernel。
- **静态量化 > 动态量化**：校准把 scale 烘成常量，省掉每个 GEMM 的 `amax→quant→dequant + device sync`（动态量化在端侧的 sync 开销极重）。

### 6.2 算子融合（计算 + 内存双收益）

融合把多个小算子合并成一个 kernel，既减少 DRAM 往返（内存维度）又减少 launch（调度维度）。

**融合的黄金法则**：

> **带宽型算子（elementwise/norm/激活/量化）尽量融进相邻算子；算力型算子（大 GEMM、attention 主体）保持独立。**
> 把 elementwise 硬塞进 GEMM 内部往往破坏 tiling、得不偿失。

典型融合模式（与具体模型无关）：
- Norm + 量化 + （cond 调制）→ 一个 kernel 产出下一 GEMM 的量化输入
- Gate + Up 两个投影合并成一个 GEMM（输出维 concat）
- 激活 + 逐元素乘 + 量化（如 SwiGLU/GeGLU）融合
- QKV 投影合并 + RoPE + 写 KV cache 融合

### 6.3 内核选型与 autotune

- **tactic autotune**：编译器在多个 kernel 实现中搜最优（针对具体 shape/架构），结果缓存复用。
- **自适应 tile**：按实际 M/N/K 规模选 tile 形状（小 batch/decode 用小 tile 防空算，大 batch/prefill 用大 tile 提吞吐）。
- **预生成 artifact**：为目标架构（如 Thor sm_110）预先编译专用 kernel，避免运行时 JIT 或 fallback 到次优 cubin。

### 6.4 NPU/BPU 类平台的计算优化特殊性

- 算子必须落在硬件**支持的算子集**内，否则回退 CPU → 性能悬崖。
- 计算优化前移到**模型结构设计**：选 BPU 高效的算子、避免不支持的动态操作。
- 量化几乎是**强制项**（INT8），且必须过厂商编译器的量化感知流程。

---

## 7. 维度三：调度优化

当瓶颈是"kernel 太多、host 跟不上、device 有空泡"（launch-bound / sync-bound）时，调度优化是唯一解——**此时量化和融合算力都帮不上忙**。

### 7.1 为什么端侧调度问题尤其突出

- 端侧 CPU 弱，host 端 Python/调度开销占比被放大。
- 单 kernel launch 开销固定（~μs 级），小算子多时累积惊人：`16000 次 launch × 2μs ≈ 33ms` 纯调度。
- 重复结构（如多步去噪循环、逐 token decode）把这个开销 ×N 倍放大。

### 7.2 调度优化手段（按力度递增）

| 手段 | 原理 | 适用 |
|------|------|------|
| **多 stream 并行** | 无依赖 kernel 并发提交 | 独立子图、IO 与计算重叠 |
| **CUDA Graph** | 把一段固定 shape 的 kernel 序列 capture 成图，一次 replay | 重复调用、固定 shape（decode/去噪循环） |
| **整图化 / 整循环融合** | 把 host 端循环下沉到 device 一次执行 | Python 逐步 dispatch 的循环 |
| **PDL 链式启动** | kernel 间在 device 上串行依赖、CPU 不参与 sync | 紧耦合 kernel 链 |
| **kernel 数压缩** | 通过融合从根本上减少 kernel 数 | launch-bound 的根治 |

### 7.3 去同步（sync-free）

device↔host 同步是隐形杀手：

- `.item()` / `.cpu()` / `print(tensor)` / boolean indexing / 动态 shape 都触发同步。
- 解法：把控制流留在 device、shape 静态化、用固定 padding 替代动态 trim、结果尽量在 device 上消费。

### 7.4 调度优化的天花板

> **编译器路径下，kernel 数由编译器决定，你改不了**。若 device 上 kernel 数过多导致 launch-bound，编译器路径会撞天花板——这时要么走手写 runtime 自己掌控 kernel 序列与启动（见 §9），要么接受这个上限。**这是判断"要不要抛弃编译器"的关键依据。**

---

## 8. 维度四：Profiling 体系

> "**没量化的优化都是迷信**"。Profiling 是前三个维度的前置与收口。

### 8.1 三个层级的 profiling

| 层级 | 回答的问题 | 工具类 | 产物 |
|------|------------|--------|------|
| **平台标定** | 这块硬件实际能跑多快？ | SysPeek 类微基准 | 实测算力/带宽/launch 开销 |
| **系统级 trace** | 端到端哪段最久、device 空不空？ | nsys 类时间线 | 时间线、kernel/内存/同步事件 |
| **kernel 级** | 这个 kernel 卡在算力还是带宽？ | ncu 类计数器 | SOL%、roofline、occupancy、warp stall |

### 8.2 自顶向下的 profiling 流程

```
平台标定（一次性，建立"屋顶"）
   │  得到 roofline 的算力屋顶 + 带宽屋顶
   ▼
系统级 trace（每轮优化）
   │  NVTX 标注各阶段 → 看哪段占比最大、device 有无空泡
   ▼
选中最大瓶颈段 → kernel 级深挖
   │  SOL%：compute-bound 还是 memory-bound？
   │  occupancy / warp stall：为什么没跑满？
   ▼
归因到维度（§4.1 四分类）→ 选手段
```

### 8.3 关键实践

- **NVTX 标注先行**：在视觉/prefill/每个循环步/采样都打 NVTX range。它不提速，但**后续所有决策都依赖这个 baseline trace**。第一周就该做。
- **roofline 归因**：算术强度（FLOP/Byte）落在带宽屋顶左侧=memory-bound，右侧=compute-bound。
- **kernel 数 vs 实算时间**：device timeline 上 kernel 总时长 ≪ 墙钟时间 → launch/sync-bound。
- **微基准对照**：用 hpc_bench 类工具单独量某个 kernel 的实测 vs 参考，确认融合/重写真的更快（编译器路径上手写融合**经常 regress**，必须实测）。
- **功耗/温度遥测**：端侧持续负载会降频，profiling 要在锁频（见 §11）后做，否则数据不可复现。

---

## 9. 两种部署范式：编译器引擎 vs 手写 runtime

这是端侧 GPU 部署最重要的战略选择，决定了"哪些优化拿得到、哪些拿不到"。

### 9.1 两种范式对比

| | **编译器引擎范式** | **手写 runtime 范式** |
|---|---|---|
| 形态 | 喂图给编译器（TRT/Myelin/TVM/天工开物），靠全局融合 + tactic 搜索 | 手写 C++/CUDA pipeline，自己掌控每个 kernel 与启动序列 |
| 优点 | 开发快、自动融合、跨 shape 鲁棒、维护成本低 | 可突破编译器天花板（kernel 数、调度、定制融合） |
| 缺点 | **kernel 数/调度被编译器锁定**，launch-bound 撞天花板 | 开发成本高、强绑定模型与硬件、维护重 |
| 适合 | compute/memory-bound 为主、结构标准的负载 | launch-bound 严重、结构固定、追求极致延迟的负载 |

### 9.2 关键判断：手写不一定更快

> **手写融合在编译器路径上经常 regress。** 编译器（如 Myelin）做全局融合，你手动插一个"自以为更优"的融合 kernel，反而制造 opaque boundary 破坏它的全局优化。
>
> 实证教训（pi05/Thor）：
> - 在编译器图里手动移 cast、改 einsum→matmul、插自定义 attention plugin —— **全部 regress**（+1~5ms 不等）。
> - 视觉编码器（SigLIP）上编译器全局融合 1.0ms **完胜**手写 6.3ms。
>
> **推论**：
> - compute/memory-bound 段 → **留给编译器**，别手痒插 plugin。
> - launch-bound 段（小算子多、调度占大头）→ 编译器拿不到那部分收益，**只有整段走手写 runtime 才能拿到**。

### 9.3 范式选择决策树

```
该段瓶颈是什么？（§4.1 归因）
   │
   ├─ compute / memory-bound ──► 编译器引擎（量化 + 让它融合）✅
   │
   └─ launch / sync-bound（kernel 数是瓶颈）
        │
        ├─ 占端到端比例小 ──► 编译器 + CUDA Graph，接受天花板
        │
        └─ 占端到端比例大 ──► 评估手写 runtime / 专用融合 kernel
                              （收益要 > 开发与维护成本）
```

> 务实策略：**大部分段用编译器，只对真正 launch-bound 且占比大的热段（如多步去噪循环）做手写**。两种范式可在同一 pipeline 内**按阶段混用**——这通常是端侧最优解。

---

## 10. 硬件架构优化

贴合具体 SoC 的内存层级与计算单元做优化。强平台绑定，收益也最大。

### 10.1 用满计算单元

- **Tensor Core / 矩阵单元**：保证 GEMM 走 Tensor Core 而非 CUDA Core；对齐 tile 形状到硬件 MMA 尺寸；喂数布局对齐（swizzle）。
- **TMA（Tensor Memory Accelerator，Hopper+/Blackwell）**：异步批量搬数据，配合 Warp Specialization（搬运 warp 与计算 warp 并行）做多 stage 流水。
- **新精度单元**：Blackwell 的 FP4 Tensor Core（~4× BF16 算力）、Thor 的 NVFP4 支持。
- **专用单元（地平线 BPU / DSP）**：把算子映射到最高效单元，避免回退通用核。

### 10.2 贴合内存层级

- **片上 SRAM/SMEM 复用**：tiling 让权重/激活块在片上多次复用，减少 global memory 访问。
- **统一内存零拷贝设计**：在 Thor/地平线上主动用 mapped/managed memory 消除阶段间逻辑拷贝（需处理 cache 一致性）。
- **带宽对齐**：访存连续、向量化（half2/packed 读写）、对齐到 cache line。

### 10.3 架构特化的注意事项

- **必须为目标架构编译**：如 Thor 是 sm_110，若 fallback 到旧架构 cubin（如 sm_101）会损失可观性能。CI 要校验产物架构。
- **版本对齐**：CUDA/JetPack/驱动/推理库版本必须与平台匹配（如 Thor JetPack 7 = CUDA 13 = cu130 wheel，不能混用桌面 cu128）。
- **新硬件特性有约束**：如 FP4 不适合 attention（q/k/v BMM 退 FP8），KV cache 用 FP8 压带宽——硬件能力要按算子类型分别评估。

---

## 11. 系统优化

最容易被忽视、但成本最低的一层。端侧"系统噪声"会让性能数据不可复现、实时性不达标。

### 11.1 电源与频率（端侧第一课）

> 端侧**持续算力 ≪ 峰值算力**。不锁频，profiling 数据随温度漂移、无法复现，实时任务会偶发掉帧。

- **功耗模式**：设为最高性能档（Jetson：`nvpmodel -m 0`）。
- **锁频**：`jetson_clocks` 锁定 GPU/CPU/EMC 到最高频，消除 DVFS 抖动。
- 量化收益评估、profiling、benchmark **都必须在锁频后做**。

### 11.2 调度与隔离（实时性）

- **CPU 亲和性**：把推理线程绑到专用核，避免被其他任务抢占。
- **IRQ/CPU 隔离 (isolcpus)**：隔离中断与系统任务，给实时推理预留干净核。
- **实时调度策略**：关键线程用 `SCHED_FIFO`/优先级，降低尾延迟抖动。

### 11.3 内存与 IO 系统配置

- **大页 (hugepage)**：减少 TLB miss（DriveOS/嵌入式收益明显）。
- **pinned/锁页内存池**：上传路径用，避免 staging。
- **预热 (warmup)**：首次执行含 JIT/分配/cache 冷启动开销，正式计时前先 warmup 数次（CUDA Graph 也需 warmup capture）。

### 11.4 端到端系统视角

- **IO 与计算重叠**：相机/传感器输入预处理与上一帧推理并行（多 stream/pipeline）。
- **批处理与流式**的取舍：端侧实时任务通常 batch 小、延迟敏感，优化目标是**单次延迟**而非吞吐。

---

## 12. 工具矩阵

| 用途 | 工具类别 | 代表工具 | 在流程中的位置 |
|------|----------|----------|----------------|
| **平台标定** | 微基准跑分 | SysPeek（实测算力/带宽/launch） | §4 步骤0、§8.1 |
| **kernel 基准** | 单算子 correctness+timing | hpc_bench（对照参考实现验证融合/重写） | §8.3 微基准对照 |
| **系统级 profiling** | 时间线 trace | nsys（Nsight Systems） | §8 系统级 |
| **kernel 级 profiling** | 硬件计数器 | ncu（Nsight Compute）、SOL%/roofline | §8 kernel 级 |
| **标注** | 代码插桩 | NVTX range | §8.3 先行项 |
| **量化** | PTQ/QAT 工具 | ModelOpt（NV）、天工开物量化（地平线） | 阶段 A |
| **图编译** | 编译器 | TensorRT、TVM、ONNX Runtime、hbdk（地平线） | 阶段 B |
| **遥测** | 功耗/温度 | tegrastats、INA3221、nvidia-smi | §11、§8.3 |
| **图分析** | 图 dump/可视化 | netron、编译器 graph dump | 阶段 A/B 调试 |

> 工具选型原则：**先有度量再优化**。最小起步集 = 一个平台标定工具（建 roofline）+ nsys（找瓶颈段）+ NVTX（标阶段）。kernel 级 ncu 在锁定热点后再上。

---

## 13. MVP 贯穿案例：π0.5 on Thor

用 π0.5 把整套方法论走一遍。π0.5 = **PaliGemma SigLIP ViT（视觉）+ Gemma LLM（prefill）+ flow-matching action expert（多步去噪）**，端侧做实时机器人动作生成。

### 13.1 负载画像与瓶颈分布

| 阶段 | 计算特征 | 瓶颈类型 | 主攻维度 |
|------|----------|----------|----------|
| 视觉 SigLIP 编码 | 大 GEMM/卷积、可 batch 多视角 | compute-bound | 计算（量化 + 编译器融合） |
| LLM prefix prefill | 长序列大 GEMM | compute-bound | 计算（FP8/NVFP4 + FMHA） |
| 多步去噪 (action expert) | 小 batch(S≈10) × N 步，小算子多 | **launch-bound** | 调度（整循环 + 融合 + CUDA Graph） |
| KV 传递 | 阶段间拷贝 | memory-bound | 内存（零拷贝/FP8 KV） |

> **关键再认知**：去噪循环是 **launch-bound** 而非 compute-bound。全网无脑上 NVFP4 是误区——NVFP4 主要帮 compute/memory-bound 的视觉/prefill；去噪的真瓶颈是 kernel 数（调度），量化收益有限。

### 13.2 按本手册流程的落地次序

```
步骤0  锁频(nvpmodel -m0 + jetson_clocks) + SysPeek 标定 Thor 真实带宽/算力
        ↓
Phase 精度先行（纯前端，低成本高收益）
        多帧分层校准 + percentile + 真实数据 recalibration
        （任务成功率可从 ~92% → ~98%）
        ↓
Phase 编译器友好性能（按 ROI）
        ① 去噪 modulation 常量预计算（只依赖步序 → 砍大量 Dense GEMM）
        ② CUDA Graph 默认开（去噪循环固定 shape）
        ③ 视觉/prefill 上 NVFP4/FP8（compute-bound 段，收益最大）
        ④ FP8 KV cache（长 prefix 带宽减半）
        ↓
Phase 异构编排
        视觉=编译器引擎（保持全局融合，勿插 plugin）
        prefill=编译器引擎（FP8 + FMHA）
        去噪循环=按需手写 runtime（launch-bound，编译器拿不到那 26ms）
        ↓
Phase 验证
        nsys 确认 CUDA Graph 真在跑、kernel 数下降
        端到端任务指标 + 延迟/功耗回归
```

### 13.3 阶段-后端异构映射（典型最优解）

| 阶段 | 选定后端 | 理由（对应本手册） |
|------|----------|---------------------|
| 视觉 SigLIP | 编译器引擎 + 多视角 batch | compute-bound，§9.2 编译器完胜手写 |
| LLM prefill | 编译器引擎 + FP8/NVFP4 KV-only | compute-bound，量化收益大 |
| 去噪循环 | 手写 runtime（整循环 + 融合 + 单 CUDA Graph） | launch-bound，§9.3 占比大→手写 |

> 实测参考量级：编译器引擎基线 ~70ms → 去噪段手写 runtime 后 ~44ms（-37%）→ NVFP4 视觉/prefill 进一步 ~40ms。**这 26ms 大头来自去噪的 launch-bound 在手写路径才拿得到，编译器路径照搬不动。**

---

## 14. 决策原则与反模式

### 14.1 八条决策原则

1. **先标定后优化**：用实测算力/带宽建 roofline，别信 datasheet 峰值。
2. **先归因后下手**：先判 compute/memory/launch/sync 四分类，再选维度，否则做无用功。
3. **精度先行**：先把校准做扎实，再激进降精度；每步过**任务级**精度 gate。
4. **ROI 排序**：先做"低风险高收益"（开开关、配置项），再碰"高风险图重写/手写"。
5. **编译器友好优先**：能让编译器融合的就别手写；手写只用于编译器拿不到的 launch-bound 热段。
6. **开关位前移**：运行时想要的能力，元数据在量化/导出阶段就写对。
7. **按阶段混范式**：编译器引擎 + 手写 runtime 在同一 pipeline 内混用，各段选最优。
8. **守住可回滚基线**：永远保留一份"已知良好"的配置。

### 14.2 反模式清单

| 反模式 | 为什么错 | 正确做法 |
|--------|----------|----------|
| 不 profiling 直接优化 | 凭直觉常优化错地方 | 先 nsys + NVTX 定位 |
| 用 datasheet 峰值算效率 | 实测常差 2-3× | SysPeek 类标定 |
| launch-bound 段上量化 | 瓶颈是 kernel 数，量化无效 | 整图化/融合/CUDA Graph |
| 给编译器友好段插手写 plugin | 破坏全局融合，regress | 留给编译器 |
| 只看 cosine 不看任务指标 | 张量相似 ≠ 任务成功 | 端到端任务指标 gate |
| 不锁频就 benchmark | 降频导致数据不可复现 | nvpmodel + jetson_clocks |
| 统一内存以为自动零拷贝 | 软件层仍有 host/device 拷贝 | 主动设计 mapped/managed |
| fallback 到非目标架构 cubin | 损失~20% 性能 | 为目标 sm 预编译 + CI 校验 |
| 无界 single-max 校准 | 被 outlier 拉爆，小值分辨率掉 | 分层 + percentile + 上包络 |

### 14.3 一句话总纲

> **标定建屋顶 → profiling 找瓶颈 → 四分类归因 → 选对维度的手段 → 精度先行地实施 → profiling 收口验证**；编译器能做的交给编译器，只对 launch-bound 的热段动手写；离硬件越近、算子越受限的平台，优化越往模型结构与量化前移。

---

## 15. 多模型并行编排与混跑性能估算

> 前面 §1-14 默认"一个模型独占设备"。但**真实端侧系统几乎都是多模型混跑**：自动驾驶域控同时跑感知（多路检测/分割/BEV）+ 预测 + 规划；机器人同时跑视觉 + VLA + 安全监控。
> 本章解决两个问题：**(A) 怎么把多个模型编排到同一块加速器上**，**(B) 怎么在真正混跑前，用单跑数据事先估算混跑性能/可行性**。

### 15.1 为什么混跑是独立的优化问题

单模型优化追求"这个模型最快"；混跑优化追求"**一组模型在共享设备上同时满足各自 SLO**"。两者目标常冲突：

- 单模型把 SM/带宽吃满 → 反而挤垮共置的其他模型。
- 端侧设备只有**一块**加速器 + 有限 LPDDR，无法像云端那样加卡。
- 实时任务有**硬截止时间**（autonomous driving 的 perception 帧、控制环），混跑引入的**抖动**比平均延迟更致命。

> 核心矛盾：**利用率 vs 可预测性**。提高 GPU 利用率（混跑）必然引入**干扰（interference）**，干扰带来不可预测性，威胁 SLO。整章都在这对矛盾里找平衡。

### 15.2 混跑的本质：共享资源争用（干扰分类）

两个 kernel 同时在 GPU 上跑，会在多个层级争用资源。**理解干扰来自哪一层，才能预测和规避。**（分类参考 ETH SoCC'25《Understanding GPU Resource Interference One Level Deeper》）

| 干扰层级 | 争用的资源 | 表现 | 关键观测指标(NCU) |
|----------|------------|------|-------------------|
| **Block 调度器** | SM 名额（block 占位） | 长 kernel 霸占 SM，短 kernel 排队（head-of-line blocking） | grid/block size vs SM 数 |
| **DRAM 带宽** | global memory 带宽 | 两个 memory-bound kernel 互相饿死 | `dram__throughput.avg.pct_of_peak` |
| **L2 cache** | 共享 L2 容量/带宽 | 互相踢出 cache 行，hit rate 暴跌 | `lts__throughput`、`lts__t_sector_hit_rate` |
| **SM 内-warp 调度** | warp scheduler（4 instr/cycle/SM 上限） | 高 IPC kernel 挤占同 SM 其他 kernel | `sm__inst_issued.avg.per_cycle_active` |
| **SM 内-计算管线** | 某类 pipeline（tensor/fp32/fp64） | 同类型算子打满同一管线 | `sm__inst_executed_pipe_*.pct_of_peak` |
| **SM 内-shared memory** | SMEM 带宽 / bank | bank conflict 拖慢共置 GEMM | `l1tex__data_pipe_lsu_wavefronts_mem_shared` |

> **重要反直觉（来自 SoCC'25 实测）**：`achieved occupancy` **不能**单独用来判断能否共置。一个 occupancy 仅 6.25% 的 kernel 可能已打满某条计算管线，共置照样 1.8-8× 变慢。**必须按"资源画像"多维判断，不能只看一个利用率数字**——这正是 Orion/Usher 等被后续工作指出的简化误区。

### 15.3 GPU 共享机制谱系（编排的"硬件/系统底座"）

从粗到细，混跑可用的隔离/共享机制：

| 机制 | 粒度 | 隔离强度 | 端侧(Jetson/Thor)可用性 | 适用 |
|------|------|----------|--------------------------|------|
| **时分（time-slicing）** | 整个上下文轮流 | 强（无并发干扰） | ✅ 默认 | 简单、不追求利用率 |
| **多 CUDA stream + 优先级** | kernel 队列 | 弱（靠硬件并发，**抢占无保证**） | ✅（但 Jetson `concurrentKernels`/抢占受限） | 端侧最常用的并发手段 |
| **MPS（Multi-Process Service）** | 进程级、可限 SM 百分比 | 中（compute 分区，**不分 DRAM**） | ⚠️ 部分 Jetson 不完整支持，需实测 | 多进程模型共置 |
| **MIG（Multi-Instance GPU）** | 硬件分区（SM+L2+DRAM 通道） | 强（物理隔离） | ❌ Jetson/Thor 一般无 MIG（数据中心卡特性） | 云端强隔离多租户 |
| **Green Contexts** | SM 子集分区 | 中强（分 SM，**不分 DRAM 通道**） | 取决于 CUDA/驱动版本 | 比 MPS 更硬的 SM 隔离 |
| **细粒度 kernel 级调度** | 单 kernel / thread block | 软件可控最细 | 需自研拦截层 | 见 §15.4 SOTA 系统 |

> **端侧关键约束**（来自 Jetson 实测/官方论坛）：
> - Jetson **缺乏硬抢占**：高优先级 stream **不保证**抢占正在跑的低优先级 kernel（CUDA 文档措辞是"may preempt"）。
> - **big.LITTLE CPU 是隐形瓶颈**：4+ 并发进程时，CPU 核不够导致 host 线程互相阻塞，吞吐不升反降（Jetson 实测：8 进程时阻塞时间 ↑70×）。
> - 没有 MIG，强隔离只能靠时分或 green context（如可用）。
> - **推论**：端侧多模型编排更依赖**软件层的主动编排（时间错峰 + 资源画像配对）**，而非硬件隔离。

#### 15.3.1 深入：弱抢占（"may preempt" 不保证）

**GPU 的执行模型：run-to-completion**

GPU 不是按"线程随时可被打断"的方式工作的（那是 CPU）。它的调度逻辑是：

```text
kernel 启动
  → block scheduler 把 thread block 映射到空闲 SM
  → 一个 block 一旦上了某个 SM，就占住该 SM 的资源
    （寄存器、shared memory、warp 槽位）
  → 直到这个 block 内所有线程跑完，才释放资源、腾出位置
```

关键点：**block 是"占位即跑到底"（run-to-completion）的**。GPU 默认不会把一个正在 SM 上执行的 block "换出去"再换回来——它是为吞吐设计的，上下文切换（几十 KB 寄存器 + shared memory）代价太大。

**CUDA stream 优先级到底改了什么**

`cudaStreamCreateWithPriority(-1)` 设的高优先级，**只影响"下一个 block 该从哪个 stream 取"这个调度决策**：

- 当某个 SM 跑完一批 block、要调度新 block 时，block scheduler **优先**从高优先级 stream 的队列里取。
- 但它**不会**把已经在 SM 上跑的低优先级 block 踢下来。

经典场景（Jetson 官方论坛实测）：

```text
低优先级大网络(300ms)先启动 → 把所有 SM 占满
高优先级小网络(50ms)后到    → 队列里排着，但 SM 全被占
                            → 只能等大网络的 block 陆续跑完腾出 SM
                            → 实测：高优小网络几乎要等到大网络快结束才动
```

设了 `-1` 高优先级，**没有发生抢占**。

**"may preempt" 这个词的分量**

NVIDIA 文档对 stream priority 的措辞是高优先级 stream **"may preempt"** 低优先级正在执行的工作——注意是 **may（可能），不是 will（一定）**：

- **硬件级抢占确实存在**（较新 compute capability 支持 instruction/block 级 preemption，主要服务于 CUDA debugger、MPS time-slicing 等），但：
  - 触发时机、粒度由硬件/驱动黑盒决定，**应用层不可控**；
  - **Jetson 这类设备限制更多**：`concurrentKernels=1`、`asyncEngineCount=1`、MPS 支持不完整，实测**优先级基本不产生抢占效果**。
- 对比 CPU：OS 调度器可以**随时**中断一个线程、保存上下文、切到另一个线程（时间片轮转 + 优先级抢占，确定性强）。GPU 默认**做不到这种确定性抢占**。

**端侧为什么致命**

实时任务（控制环、感知帧）的本质诉求是"**我来了就得尽快插队**"。但 GPU 没有可靠抢占，意味着：

> 一个低优先级长 kernel 一旦占满 GPU，你的高优先级实时 kernel **没有硬件手段把它赶下来**，只能干等。这就是混跑下尾延迟（p99）爆炸的根因之一——**head-of-line blocking（队头阻塞）**。

**既然不能靠抢占，只能靠这些（软件层规避）**

| 手段 | 原理 |
|------|------|
| **kernel slicing（切片）** | 把大 kernel 切成很多小 kernel → 每个 block "占位"时间短 → 高优任务有更多"插入点"（Tally 的核心做法） |
| **时间错峰** | 编排上别让长 kernel 和实时任务同时触发 |
| **SM 分区** | 给低优任务限制可用 SM（MPS 百分比 / green context），物理上给高优留资源 |
| **整图化 + 短 kernel** | 单模型先做调度优化（§7），kernel 越短越可预测，越好"插队" |

#### 15.3.2 深入：big.LITTLE CPU（隐形瓶颈）

**big.LITTLE 是什么**

ARM 嵌入式/移动 SoC（Jetson、Thor 都是 ARM）的 CPU 是**异构多核**：

```text
┌─────────────────────────────────────┐
│  big 大核（高频高性能、费电）  × 少数   │  ← 重负载、需要算力的线程
│  LITTLE 小核（低频省电、性能弱）× 若干  │  ← 轻负载、后台任务
└─────────────────────────────────────┘
       OS 调度器（EAS 能效感知调度）决定线程放哪类核
```

设计目的是**省电**：轻活给小核，重活才上大核。线程跑在大核还是小核，由 OS 动态决定。

**为什么 GPU 推理高度依赖 CPU**

很多人以为"推理在 GPU 上，CPU 没事干"——错。**CPU 是 GPU 的"喂食者"**，host 端工作量很大：

- **kernel launch**：每个 kernel 都要 CPU 去发起（launch），小算子多的模型，CPU 要疯狂提交；
- **框架/Python 调度**：算子分发、tensor 元数据管理；
- **内存管理**：分配/释放、H2D/D2H 拷贝的发起；
- **stream 同步**：等待、回调；
- **数据预处理**：图像 resize/归一化等（常在 CPU）。

**每一个推理进程/线程，都需要至少一个 CPU 核持续、及时地喂 GPU。** 如果 CPU 喂不过来，GPU 就空泡——这就是 launch-bound 的 host 侧根源。

**为什么并发多模型时它成了瓶颈**

矛盾出在"**进程数 > 可用大核数**"：

```text
4 个推理进程并发，但只有 2-3 个大核能跑重负载：
  → 多个 host 线程抢有限大核
  → OS 被迫在它们之间时分切换 → 上下文切换 + 调度延迟
  → 部分线程被甩到慢的小核 → 喂 GPU 速度骤降 → GPU 反而闲着
  → 结果：并发越多，吞吐不升反降
```

Jetson 实测（ISPASS'25 并发 vision 推理 profiling）：

> Orin Nano 上只有少数核能承载重负载；并发进程从 1-2 个增到 **4 个、8 个**时，host 端 **blocking time 暴涨约 30×、70×**，吞吐因调度低效**不升反降**。

这就是"**隐形瓶颈**"：你 nsys 一看 GPU 利用率不高，以为还能再塞模型，实际是 **CPU 大核不够，host 端根本喂不动 GPU**，瓶颈在 CPU 不在 GPU。

**怎么应对**

| 手段 | 作用 |
|------|------|
| **CPU 亲和性绑核（affinity）** | 把关键推理线程**钉死在指定大核**，不让 OS 把它甩到小核或频繁迁移 |
| **isolcpus / cpuset 隔离** | 给实时推理**预留专用大核**，系统任务和中断赶到别的核，消除抢占 |
| **jetson_clocks 锁频** | 把 CPU（含小核）锁到最高频，减少 DVFS 抖动和小核拖累 |
| **CUDA Graph / 整图化** | 把"一长串 kernel launch"压成"一次 replay"，**大幅减少 CPU 的 launch 工作量**——直接缓解 host 瓶颈 |
| **用线程而非多进程** | 官方建议 "single process + multiple threads"，减少进程级开销与上下文 |
| **控制并发度** | 并发推理任务数 **≤ 可用大核数**，别超订阅 CPU |

> **CUDA Graph 在这里是双重收益**：既减少 GPU 端 launch 调度（§7），又减少 CPU 端的 launch 提交压力（big.LITTLE 瓶颈缓解）。这是端侧多模型场景下它优先级很高的原因。

**两者联系：端侧混跑的两条硬约束**

| 约束 | 表现 | 应对方向 |
|------|------|----------|
| **GPU 弱抢占** | 长 kernel 占满 SM，高优任务无法插队 | kernel 切片、时间错峰、SM 分区 |
| **CPU big.LITTLE** | host 喂不动 GPU，利用率虚低 | 绑核、CUDA Graph、控制并发度 |

> 端侧既没有可靠的 GPU 硬抢占，又有羸弱/异构的 CPU。所以"提高利用率"不能靠硬件自动搞定，必须靠**主动软件编排**——CUDA Graph 减少 host 压力、绑核避免 CPU 争抢、kernel 切片 + 时间错峰弥补无抢占、SM 分区做软隔离。这正是 §15.3 那句"端侧更依赖软件层主动编排，而非硬件隔离"的底层原因。

### 15.4 业界先进的细粒度编排系统（SOTA 参考）

当 MPS/MIG 不够用时，研究界用**拦截 kernel launch + 细粒度调度**来做干扰感知共置。核心思想可借鉴到端侧自研 runtime。

| 系统 | 核心机制 | 关键思想（可借鉴点） |
|------|----------|---------------------|
| **Orion** (EuroSys'24) | 拦截 kernel launch，按**算子大小 + compute/memory-bound 属性**调度 | **把 compute-bound 与 memory-bound 算子配对共置**，利用率↑最高 7.3×、高优 p99 仅 +15% |
| **Tally** (2024) | LD_PRELOAD 拦截，**kernel slicing + preemption**，block 级调度 | 大 kernel 切片 + 抢占，给高优任务做**非侵入式性能隔离** |
| **IasRT** (ICCD'25) | kernel 级 profiling + **动态 SM 分区** + SLO 控制器 | 多优先级实时推理，p99 ↓最高 38% |
| **gpulets** | MPS 分区 + 用 **L2/DRAM/SM 吞吐**做干扰预测特征 | 离线 NCU 画像 → 预测共置 slowdown |
| **DARIS** (2025) | MPS + stream 空分 + 同步分段时分，**超额订阅** | 实时优先级 + 零延迟分区迁移，高优全部满足截止时间 |
| **SKADI** | block 级特征 → **零样本预测未见模型延迟** | 不必每个新模型都重测，泛化预测 |
| **REEF / Clockwork** | 抢占式（REEF, AMD）/ 高度可预测执行（Clockwork） | 实时性 / 可预测性范式 |

> 共同范式：**(1) 细粒度（per-kernel/per-block）调度避免 head-of-line blocking；(2) 用资源画像预测干扰、挑"互补"的负载配对；(3) 用 SM 分区/优先级在利用率与 SLO 间动态平衡。**

### 15.5 事先估算：从单跑数据预测混跑性能

> 这是用户最关心的部分：**不真正混跑，怎么估算混跑会不会崩、能塞几个模型、谁和谁配。** 分三个层次，由粗到精、由便宜到准。

#### 层次 1：资源预算法（最便宜，先做可行性筛查）

把每个模型单跑时的**资源需求**加总，与平台峰值比，判断"会不会撞墙"。

```
对每个模型 i 单跑测：
  - 平均/峰值 DRAM 带宽需求  BW_i      (GB/s, 由 SysPeek 标定峰值 + nsys/NCU 实测占用)
  - 平均 SM 占用             SM_i      (%)
  - 单跑延迟                 T_i        (ms)
  - 显存常驻                 M_i        (权重+KV+激活峰值)

可行性快筛（必要非充分条件）：
  ① 显存:  Σ M_i  <  设备可用内存            ← 不满足直接 OOM，先砍
  ② 带宽:  Σ BW_i 接近/超过 峰值带宽?        ← memory-bound 叠加 → 必然互相拖慢
  ③ 算力:  Σ (compute-bound 模型的算力需求) > 峰值? ← compute 饱和
```

- **带宽屋顶是端侧最先撞的墙**：LPDDR 带宽有限（Thor ~273 GB/s），多个 memory-bound 模型（如多个 decode/小算子模型）叠加，`Σ BW_i` 一旦接近峰值，每个模型都按比例变慢。
- 这一层**只筛掉明显不可行的组合**，不给精确延迟。

#### 层次 2：roofline + 互补性配对（挑"谁和谁混"）

按 roofline 把每个模型（甚至每个 kernel）分类，**让互补的负载配对**：

| 模型 A | 模型 B | 共置预期 |
|--------|--------|----------|
| compute-bound | memory-bound | ✅ **最佳**：争用不同资源，接近"免费"并发（Orion 核心洞察） |
| memory-bound | memory-bound | ❌ 抢 DRAM 带宽，双双变慢 |
| compute-bound | compute-bound | ❌ 抢 Tensor Core/管线，双双变慢 |
| 小 launch-bound | 任意 | ◐ 看 SM 空泡能否被填，但小 kernel 多易触发调度干扰 |

> 实操：用 NCU 的 roofline 对每个模型出一张"资源画像雷达图"（SM 利用、DRAM 利用、L2、各 pipe）。**画像互补 → 适合共置；画像重叠 → 避免或时分错峰。**

#### 层次 3：干扰敏感度模型（最准，量化 slowdown）

ETH SoCC'25 提出的方法论：**为每个 kernel 建"对每类资源的敏感度曲线"**，再合成 workload 级预测。

```
离线（每个模型单跑，一次性）：
  对每个 kernel 测两组量——
    (a) 资源「需求」：它占了多少 DRAM BW / SM / L2 / 各 pipe（NCU 画像）
    (b) 资源「敏感度」：人为施加该资源压力（microbenchmark 注入争用），
        测它变慢多少 → 得到 slowdown(资源, 压力) 曲线

在线预测共置 (A, B)：
  对时间上重叠的 kernel 对 (a_k, b_k)：
    pressure_on_A = B 在每类资源上的「需求」
    slowdown_A_k  = A 的敏感度曲线在该压力下的值
  workload 级 slowdown_A ≈ 各重叠 kernel slowdown 的（按时间加权）合成
  预测 T_A' = T_A × slowdown_A，与 SLO 比
```

要点：
- **kernel 级 → workload 级**：先预测每个 kernel 的 slowdown，再按执行时间加权合成整个模型的 slowdown（避免黑盒 ML 模型不可解释、不可迁移的问题）。
- **微基准注入**是关键工具：用可调压力的 microbenchmark（专打 DRAM / L2 / SMEM / 某 pipe）测敏感度，比直接两两实测组合数少得多（N 个模型只需各测一次，而非 N² 次配对）。
- **泛化（SKADI 思路）**：用 block 级特征做回归，可对**未实测过的新模型**零样本预测，省去每次上新模型都重测。
- **必须按目标 GPU 重测**：cuBLAS/cuDNN 在不同架构实现不同，Thor 的画像不能套用桌面卡（SoCC'25 明确指出）。

#### 估算方法选型

| 你的问题 | 用哪层 | 成本 |
|----------|--------|------|
| 这组模型会不会 OOM / 撞带宽墙？ | 层次 1 资源预算 | 极低（单跑指标加总） |
| 这 N 个模型，谁和谁配着跑好？ | 层次 2 roofline 配对 | 低（每模型一张画像） |
| 共置后高优模型 p99 会涨多少、还满足 SLO 吗？ | 层次 3 敏感度模型 | 中（NCU 画像 + 微基准） |

### 15.6 编排优化策略（估算之后怎么排）

| 策略 | 做法 | 适用 |
|------|------|------|
| **互补配对共置** | compute-bound × memory-bound 放一起（§15.5 层次2） | 有空闲资源、想提利用率 |
| **优先级分级** | 关键实时任务（控制/感知）高优 stream；尽力而为任务（日志/录制）低优 | 有明确 SLO 分级 |
| **时间错峰** | 让重负载模型在时间上交错触发，避开同时打满 | 周期性任务（多路相机不同帧率） |
| **SM 空间分区** | MPS `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` / green context 给各模型划 SM | 干扰严重、需硬隔离且平台支持 |
| **整图化 + 短 kernel** | 每个模型先做单模型调度优化（§7），kernel 越短越好切片/错峰 | 降低 head-of-line blocking |
| **降配换确定性** | 给单个模型限 SM（如 decode 限 < 半数 SM），换整体可预测性 | 实时系统优先保 SLO 而非单模型峰值 |
| **CPU/IO 协同** | 绑核（§11.2）避免 big.LITTLE 上 host 线程互相阻塞 | 端侧多进程并发必做 |

> 端侧务实顺序：**先单模型各自优化好（§5-9）→ 资源预算快筛（层次1）→ roofline 配对（层次2）→ 优先级 + 时间错峰编排 → 实测验证 → 必要时上 SM 分区/敏感度模型（层次3）**。

### 15.7 混跑的评估指标与方法论

单模型看"延迟"，混跑必须看一组新指标：

| 指标 | 含义 | 为什么重要 |
|------|------|-----------|
| **per-model p99 / 尾延迟** | 每个模型在混跑下的 99 分位延迟 | 实时系统的命门是尾延迟，不是平均 |
| **SLO 满足率 / 截止时间命中率** | 高优任务按时完成的比例 | 端侧硬实时的直接 KPI |
| **slowdown（混跑/单跑）** | 每个模型相对独占的变慢倍数 | 量化干扰代价 |
| **aggregate throughput** | 整设备总吞吐 | 利用率收益 |
| **抖动（jitter / 方差）** | 延迟波动 | 控制环对抖动比对均值更敏感 |
| **功耗/温度** | 混跑下的持续功耗 | 端侧可能触发降频，反噬性能 |

> 评估方法：在**锁频**（§11.1）下，用 nsys 抓混跑时间线确认 kernel 是否真并发（还是被串行化）、有无 head-of-line blocking；按"**先保高优 SLO，再看总吞吐**"的次序判定一个编排方案是否可接受。

### 15.8 一句话总纲（多模型）

> **混跑的本质是共享资源争用；编排的目标是在"利用率 vs 可预测性"间取舍。事先估算分三层——资源预算快筛（会不会撞墙）→ roofline 互补配对（谁和谁混）→ 干扰敏感度模型（slowdown 多少）；端侧因缺 MIG/弱抢占/弱 CPU，更要靠软件层的优先级 + 时间错峰 + 互补配对，而非硬件隔离。**

---

## 附：与其他文档的关系

本手册是**顶层方法论总纲**。各维度/平台的具体实现细节、实测数据、代码落点，见对应的专题文档（量化原理、kernel 融合设计、KV 链路、校准策略、平台标定报告等）。本文档只负责给出"框架与次序"，不复制实现细节。

