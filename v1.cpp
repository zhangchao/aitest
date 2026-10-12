/**
 * code2/v1_baseline_real/colbert_maxsim_v1.cpp
 *
 * 目标：理解+重整用户提供的"目标机器 CPU 版本可跑参考"
 * (CPU上能跑的代码.docx)，产出一份语义完全等价、但结构按模块重新
 * 组织、注释讲清楚每一步对应 design.md 哪个概念的真实实现。
 *
 * 这份代码本身是对参考代码的忠实重写（算法、API 调用、tiling 策略
 * 全部和参考一致），不是我瞎编的新设计 —— 它的价值在于它用的是
 * **真实 AscendC API**（Matmul 模板类 + REGIST_MATMUL_OBJ + ReduceMax
 * 的 Pattern::Reduce::AR 模式），不是 code/ascendc_colbert_maxsim 里
 * 手写占位的三重循环。
 *
 * 和参考代码一致的关键设计决策（见文末 README 的优化方案分析）：
 *   1. 并行粒度 = batch：host 侧 for 循环对每个 batch 启动一次独立的
 *      kernel（<<<1, ...>>>），单个 kernel 内 SetDim(1) 只用 1 个核。
 *   2. Cube 分块（baseM/baseN）、L1/L0 双缓冲，全部交给
 *      MultiCoreMatmulTiling + Matmul 模板类内部处理，host/kernel 代码
 *      不手写 K 轴循环和 TQue 双缓冲队列。
 *   3. ReduceMax 直接对 Cube 输出的 (baseM,baseN) 分块做行方向规约，
 *      在线 Max 合并到 maxBuf，循环跑完 Matmul 的所有 Iterate() 分块。
 *   4. ReduceSum 在同一个核内对整个 M 轴一次性求和，没有跨核合并问题
 *      （因为一个 batch 从头到尾只在一个核上处理）。
 *
 * 编译方式：和原参考一致，需要在 910B3 容器内
 *   source /usr/local/Ascend/ascend-toolkit/set_env.sh
 *   bisheng ... colbert_maxsim_v1.cpp -o colbert_maxsim_v1 ...
 * （具体编译命令以 /usr/local/Ascend/ascend-toolkit/latest/samples/operator/
 *  下同类单文件 Matmul+Vector 融合算子样例的 Makefile 为准）
 * 本机（mac，无 CANN 工具链）无法编译验证，这是明确告知的限制，不是
 * 隐瞒的问题。
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul;

#define CHECK_ACL(expr)                                                                                 \
    do {                                                                                                \
        auto __ret = (expr);                                                                            \
        int32_t __code = static_cast<int32_t>(__ret);                                                   \
        if (__code != 0) {                                                                              \
            std::fprintf(stderr, "[ERROR] %s failed at %s:%d, ret=%d\n", #expr, __FILE__, __LINE__,    \
                         __code);                                                                       \
            return 1;                                                                                   \
        }                                                                                               \
    } while (0)

// 选一个能整除 M/N 且 <=128 的 16 倍数作为 Cube 分块基准（baseM/baseN）。
// 对应 design.md 里"Cube 以 16x16x16 为粒度做矩阵乘"——这里把切分粒度
// 对齐到 16 的倍数，交给硬件 Cube 单元高效处理。
static int32_t PickBase(uint32_t dim)
{
    for (int32_t bm = 128; bm >= 16; bm -= 16) {
        if (dim % bm == 0) return bm;
    }
    return (int32_t)dim;
}

// 用官方 MultiCoreMatmulTiling 生成 Cube 侧 TCubeTiling —— 这是"交给
// 框架做 tiling"的正确姿势，不需要像 code/ascendc_colbert_maxsim 里那样
// 手写 kBlock/mBlock 的 UB 容量反推逻辑（那是另一层、Vector 侧的 tiling，
// 两者不冲突，见 v2_optimized 里怎么把两者结合）。
AscendC::tiling::TCubeTiling GenerateTiling(platform_ascendc::PlatformAscendC* ascendcPlatform,
                                            uint32_t M, uint32_t K, uint32_t N)
{
    const int32_t baseM = PickBase(M);
    const int32_t baseN = PickBase(N);

    matmul_tiling::MultiCoreMatmulTiling cubeTiling(*ascendcPlatform);
    cubeTiling.SetDim(1);  // 单 batch 单核：并行粒度在 host 侧的 batch 循环，不在这里
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetCType(matmul_tiling::TPosition::VECIN, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);
    cubeTiling.SetFixSplit(baseM, baseN, -1);
    cubeTiling.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    AscendC::tiling::TCubeTiling cubeTilingData;
    if (cubeTiling.GetTiling(cubeTilingData) == -1) {
        std::fprintf(stderr, "Generate matmul tiling failed.\n");
    }
    return cubeTilingData;
}

// ================= 核函数：单 batch [M,K]x[K,N] -> 行max -> 求和 -> out[0] =================
// 对应 ColBERT MaxSim: score = sum_m max_n (Q[m,:] . Doc[n,:])
// 融合点（design.md "融合算子"的核心）：Matmul 输出的 (baseM,baseN) 分块
// 从头到尾只活在 UB/VECIN 里，ReduceMax 直接消费，从不写回 GM —— 消灭了
// 朴素三算子版本里 BatchMatMul 输出整个 (M,N) 分数矩阵到 GM、再读回来做
// ReduceMax 的那次完整 GM 读写往返。
extern "C" __global__ __mix__(1, 2) void BatchMatmulMaxSumKernel(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* out,
    __gm__ uint8_t* __kfc_workspace__ workspace, AscendC::tiling::TCubeTiling tiling)
{
    AscendC::TPipe pipe;

    using A_T = half;    // Q/Doc 用 fp16 —— 真机上 Cube 的标准输入精度
    using B_T = half;
    using C_T = float;   // Cube 输出 PSUM 累加用 fp32，避免精度损失

    const uint32_t M = tiling.M;
    const uint32_t N = tiling.N;
    const uint32_t Ka = tiling.Ka;
    const uint32_t baseM = tiling.baseM;
    const uint32_t baseN = tiling.baseN;

    AscendC::GlobalTensor<A_T> aGlobal;
    AscendC::GlobalTensor<B_T> bGlobal;
    AscendC::GlobalTensor<C_T> outGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(a), M * Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(b), Ka * N);
    outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(out), 1);

    // Matmul 模板对象：A/B 在 GM，C 直接落在 VECIN（给 Vector 用，不回 GM）
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T>,
           MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, C_T>>
        matmulObj;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, &tiling);

    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> cQueue;
    pipe.InitBuffer(cQueue, 1, baseM * baseN * sizeof(C_T));

    AscendC::TBuf<AscendC::QuePosition::VECOUT> rowMaxBuf, maxBufBuf, rowSumBuf;
    pipe.InitBuffer(rowMaxBuf, baseM * sizeof(C_T));
    pipe.InitBuffer(maxBufBuf, baseM * sizeof(C_T));
    pipe.InitBuffer(rowSumBuf, sizeof(C_T));

    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);

    LocalTensor<C_T> maxBuf = maxBufBuf.Get<C_T>();
    AscendC::Duplicate(maxBuf, (C_T)-1e30f, baseM);   // running max 初值 -inf

    // Matmul 的 Iterate() 内部按 baseM x baseN 分块遍历整个 (M,N)，
    // 每吐出一块就立即做 ReduceMax + 在线合并 —— 这就是"在线 Max 合并"
    // （对应 code/step2_fused_colbert.py 的 running_max 逻辑），区别是
    // 这里分块边界是 N 轴（doc token 维度），而不是 step2 里显式的 K 轴块。
    while (matmulObj.template Iterate<true>()) {
        LocalTensor<C_T> cLocal = cQueue.AllocTensor<C_T>();
        matmulObj.template GetTensorC<true>(cLocal, false, true);
        cQueue.EnQue(cLocal);
        cLocal = cQueue.DeQue<C_T>();

        LocalTensor<C_T> rowMax = rowMaxBuf.Get<C_T>();
        uint32_t reduceShape2D[] = {baseM, baseN};
        ReduceMax<C_T, AscendC::Pattern::Reduce::AR>(rowMax, cLocal, reduceShape2D, false);
        AscendC::Max(maxBuf, maxBuf, rowMax, baseM);

        cQueue.FreeTensor(cLocal);
    }
    matmulObj.End();

    // 单核内整段 M 轴一次性 ReduceSum —— 没有跨核合并问题，
    // 因为这个 batch 从头到尾只分配给了这一个核（SetDim(1) + host 按
    // batch 发射 kernel 的并行策略决定的）。
    LocalTensor<C_T> rowSum = rowSumBuf.Get<C_T>();
    uint32_t reduceShape1D[] = {1, baseM};
    ReduceSum<C_T, AscendC::Pattern::Reduce::AR>(rowSum, maxBuf, reduceShape1D, false);
    AscendC::DataCopy(outGlobal[0], rowSum, 1);
}

// ================= host =================
int32_t main(int32_t argc, char** argv)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();

    constexpr uint32_t batch = 2;
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 64;
    constexpr uint32_t N = 64;
    const size_t aSize = batch * M * K * sizeof(aclFloat16);
    const size_t bSize = batch * K * N * sizeof(aclFloat16);
    const size_t outSize = batch * sizeof(float);
    const size_t workspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    auto tiling = GenerateTiling(ascendcPlatform, M, K, N);

    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t* aDevice = nullptr;
    uint8_t* bDevice = nullptr;
    uint8_t* outDevice = nullptr;
    uint8_t* workspaceDevice = nullptr;
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void**>(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void**>(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void**>(&outDevice), outSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(reinterpret_cast<void**>(&workspaceDevice), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    std::vector<aclFloat16> aHost(batch * M * K);
    std::vector<aclFloat16> bHost(batch * K * N);
    std::vector<float> outHost(batch, 0.0f);
    std::vector<float> golden(batch, 0.0f);

    srand(42);
    for (size_t i = 0; i < aHost.size(); i++) aHost[i] = aclFloatToFloat16((float)(rand() % 100) / 100.0f);
    for (size_t i = 0; i < bHost.size(); i++) bHost[i] = aclFloatToFloat16((float)(rand() % 100) / 100.0f);

    // golden：y[b] = sum_m max_n (A[b,m,:] . B[b,:,n])
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t m = 0; m < M; m++) {
            float rowMax = -1e30f;
            for (uint32_t n = 0; n < N; n++) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < K; k++) {
                    acc += aclFloat16ToFloat(aHost[b * M * K + m * K + k]) *
                           aclFloat16ToFloat(bHost[b * K * N + k * N + n]);
                }
                rowMax = std::max(rowMax, acc);
            }
            golden[b] += rowMax;
        }
    }

    CHECK_ACL(aclrtMemcpy(aDevice, aSize, aHost.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(bDevice, bSize, bHost.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // 并行粒度 = batch：每个 batch 一次独立 kernel 发射。
    // 910B3 有多个 AI Core，batch 数够大时这些 launch 会被调度到不同核，
    // 天然并行；batch 数很小（比如本例 batch=2）时大量核会闲置——
    // 这正是 v2_optimized 要解决的问题，见 code2/README.md。
    for (uint32_t b = 0; b < batch; b++) {
        BatchMatmulMaxSumKernel<<<1, 0, stream>>>(
            aDevice + b * M * K * sizeof(aclFloat16),
            bDevice + b * K * N * sizeof(aclFloat16),
            outDevice + b * sizeof(float),
            workspaceDevice, tiling);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(outHost.data(), outSize, outDevice, outSize, ACL_MEMCPY_DEVICE_TO_HOST));

    std::printf("Output: ");
    for (uint32_t i = 0; i < batch; i++) std::printf("%.2f ", outHost[i]);
    std::printf("\nGolden: ");
    for (uint32_t i = 0; i < batch; i++) std::printf("%.2f ", golden[i]);
    std::printf("\n");

    bool pass = true;
    for (uint32_t i = 0; i < batch; i++) {
        if (std::fabs(outHost[i] - golden[i]) > 0.05f) pass = false;
    }
    std::printf("test %s\n", pass ? "pass" : "failed");
    if (!pass) return 1;

    CHECK_ACL(aclrtFree(aDevice));
    CHECK_ACL(aclrtFree(bDevice));
    CHECK_ACL(aclrtFree(outDevice));
    if (workspaceDevice != nullptr) CHECK_ACL(aclrtFree(workspaceDevice));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}
