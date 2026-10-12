/**
 * code2/v2_optimized/colbert_maxsim_v2.cpp
 *
 * v1_baseline_real/colbert_maxsim_v1.cpp 的单文件版本——不再拆
 * op_host/op_kernel/test 三个文件，方便整段复制到容器里编译跑。
 * 内容和之前拆开的三个文件（colbert_maxsim_tiling_v2.h/.cpp +
 * colbert_maxsim_kernel_v2.cpp）完全一致，只是合并了物理位置。
 *
 * 和 v1 的核心区别（详见 ../README.md 的优化方案分析表）：
 *   v1：SetDim(1)，host 侧 for(b) 循环逐 batch 发射 kernel，并行粒度=batch。
 *   v2：(B,M) 联合切分，一次性发射 coreNum 个核，GetBlockIdx() 轮询消费
 *       全部任务块，核内显式 K 轴双缓冲，跨核用 SetAtomicAdd 合并同一
 *       batch 的不同 M 子块——这是 design.md 要求、v1（docx 参考）没做
 *       的多核优化。
 *
 * 编译方式同 v1：910B3 容器内 source set_env.sh 后用 bisheng 编译，
 * 具体命令参照 samples/operator/ 下同类单文件样例的 Makefile。
 * 本机（mac，无 CANN 工具链）无法编译验证，头文件缺失是预期现象。
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
using namespace AscendC;

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

namespace {
constexpr int32_t BUFFER_NUM = 2;  // doc 分块双缓冲：Ping + Pong

int32_t PickBase(uint32_t dim)
{
    for (int32_t bm = 128; bm >= 16; bm -= 16) {
        if (dim % bm == 0) return bm;
    }
    return (int32_t)dim;
}

uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
}  // namespace

// ================= tiling 数据结构（原 colbert_maxsim_tiling_v2.h）=================
struct ColbertMaxSimTilingV2 {
    uint32_t B;
    uint32_t M;
    uint32_t K;      // doc token 数
    uint32_t D;      // 向量维度

    uint32_t baseM;  // Cube 分块 M（16 对齐）
    uint32_t baseK;  // Cube 分块 K（16 对齐，doc token 轴）

    uint32_t mTile;          // 多核调度层 M 分块大小
    uint32_t mTilesPerBatch;
    uint32_t totalBlocks;    // B * mTilesPerBatch
    uint32_t coreNum;        // min(物理核数, totalBlocks)
};

// ================= tiling 计算（原 colbert_maxsim_tiling_v2.cpp）=================
// 两层 tiling：Cube 层(baseM/baseK)照搬 v1 验证过的 PickBase 套路；
// 调度层(mTile/totalBlocks/coreNum)是 v2 新加的 (B,M) 联合切分逻辑。
ColbertMaxSimTilingV2 ComputeColbertMaxSimTilingV2(uint32_t B, uint32_t M, uint32_t K, uint32_t D,
                                                    uint32_t physicalCoreNum)
{
    ColbertMaxSimTilingV2 t{};
    t.B = B;
    t.M = M;
    t.K = K;
    t.D = D;

    t.baseM = (uint32_t)PickBase(M);
    t.baseK = (uint32_t)PickBase(K);

    // mTile 必须整除 M，否则最后一个 M 子块（尾块）的实际 mLen 会比
    // GenerateCubeTiling 里固定死的 mTile 小，但 cubeTiling 的 shape 是
    // 全局生成一次、对所有块复用的——尾块传真实 mLen 进 Matmul 会跟
    // 已经编译进 cubeTiling 的 baseM/shape 对不上。这里用和 baseK 一样
    // 的"找一个能整除的候选值，找不到就退化成不切块"策略，彻底消灭
    // 这类尾块不对齐问题，而不是留到运行时才发现算错。
    uint32_t mTile = t.baseM * 2;
    if (mTile > M) {
        mTile = M;
    } else {
        bool found = false;
        for (uint32_t cand = mTile; cand >= t.baseM; cand -= t.baseM) {
            if (M % cand == 0) {
                mTile = cand;
                found = true;
                break;
            }
        }
        if (!found) mTile = M;  // 没有整除的候选，退化成每个 batch 一个块（无尾块）
    }
    t.mTile = mTile;
    t.mTilesPerBatch = CeilDiv(M, mTile);
    t.totalBlocks = B * t.mTilesPerBatch;
    // 核数不超过任务数——batch/M 很小时没必要占满全部 AI Core
    t.coreNum = (physicalCoreNum < t.totalBlocks) ? physicalCoreNum : t.totalBlocks;
    return t;
}

AscendC::tiling::TCubeTiling GenerateCubeTiling(platform_ascendc::PlatformAscendC *ascendcPlatform,
                                                 uint32_t mTile, uint32_t D, uint32_t baseK)
{
    const int32_t baseM = PickBase(mTile);
    const int32_t baseN = (int32_t)baseK;

    matmul_tiling::MultiCoreMatmulTiling cubeTiling(*ascendcPlatform);
    cubeTiling.SetDim(1);  // 每个 AI Core 内部仍是单核 Cube tiling，跨核并行靠外层 GetBlockIdx()
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetBType(matmul_tiling::TPosition::VECIN, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetCType(matmul_tiling::TPosition::VECIN, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(mTile, baseK, D);
    cubeTiling.SetOrgShape(mTile, baseK, D);
    cubeTiling.SetFixSplit(baseM, baseN, -1);
    cubeTiling.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    AscendC::tiling::TCubeTiling cubeTilingData;
    if (cubeTiling.GetTiling(cubeTilingData) == -1) {
        std::fprintf(stderr, "Generate matmul tiling failed.\n");
    }
    return cubeTilingData;
}

// ================= Device 侧 Kernel（原 colbert_maxsim_kernel_v2.cpp）=================
class KernelColbertMaxSimV2 {
public:
    __aicore__ inline KernelColbertMaxSimV2() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR doc, GM_ADDR score,
                                 const ColbertMaxSimTilingV2 &tiling,
                                 const AscendC::tiling::TCubeTiling &cubeTiling) {
        tiling_ = tiling;
        cubeTiling_ = cubeTiling;

        qGm_.SetGlobalBuffer((__gm__ half *)q, tiling_.B * tiling_.M * tiling_.D);
        docGm_.SetGlobalBuffer((__gm__ half *)doc, tiling_.B * tiling_.K * tiling_.D);
        scoreGm_.SetGlobalBuffer((__gm__ float *)score, tiling_.B);

        pipe_.InitBuffer(docQueue_, BUFFER_NUM, tiling_.baseK * tiling_.D * sizeof(half));
        pipe_.InitBuffer(cQueue_, BUFFER_NUM, tiling_.mTile * tiling_.baseK * sizeof(float));
        pipe_.InitBuffer(rowMaxBuf_, tiling_.mTile * sizeof(float));
        pipe_.InitBuffer(runningMaxBuf_, tiling_.mTile * sizeof(float));
        pipe_.InitBuffer(partialSumBuf_, sizeof(float));

        REGIST_MATMUL_OBJ(&pipe_, GetSysWorkSpacePtr(), matmulObj_, &cubeTiling_);
    }

    __aicore__ inline void Process() {
        // (B,M) 联合切分，轮询分发到各核——v1 按 batch 发射 kernel 做不到
        // 的多核并行层，这里是 v2 新增的核心优化。
        for (uint32_t idx = GetBlockIdx(); idx < tiling_.totalBlocks; idx += tiling_.coreNum) {
            uint32_t b = idx / tiling_.mTilesPerBatch;
            uint32_t mTileIdx = idx % tiling_.mTilesPerBatch;
            uint32_t mStart = mTileIdx * tiling_.mTile;
            uint32_t mLen = (mStart + tiling_.mTile <= tiling_.M)
                                 ? tiling_.mTile
                                 : (tiling_.M - mStart);
            ProcessOneBlock(b, mStart, mLen);
        }
    }

private:
    __aicore__ inline void ProcessOneBlock(uint32_t b, uint32_t mStart, uint32_t mLen) {
        LocalTensor<float> runningMax = runningMaxBuf_.Get<float>();
        Duplicate(runningMax, -3.0e38f, mLen);

        // 显式 K 轴（doc token 轴）分块：v1 里这层由 Matmul::Iterate()
        // 隐式管理，v2 为了配合外层 (B,M) 多核切分，必须显式拿出来控制，
        // 否则没法按子块独立发给不同核处理。
        for (uint32_t kStart = 0; kStart < tiling_.K; kStart += tiling_.baseK) {
            uint32_t kLen = (kStart + tiling_.baseK <= tiling_.K)
                                 ? tiling_.baseK
                                 : (tiling_.K - kStart);

            // ---- CopyIn doc 分块，双缓冲由 docQueue_ 的 TQue 机制隐藏 ----
            LocalTensor<half> docLocal = docQueue_.AllocTensor<half>();
            DataCopy(docLocal, docGm_[(b * tiling_.K + kStart) * tiling_.D], kLen * tiling_.D);
            docQueue_.EnQue(docLocal);
            docLocal = docQueue_.DeQue<half>();

            // ---- Cube：复用 v1 验证过的真实 Matmul 调用方式 ----
            matmulObj_.SetTensorA(qGm_[(b * tiling_.M + mStart) * tiling_.D]);
            matmulObj_.SetTensorB(docLocal, /*isTransB=*/true);

            LocalTensor<float> cLocal = cQueue_.AllocTensor<float>();
            matmulObj_.template IterateAll<true>(cLocal);
            cQueue_.EnQue(cLocal);
            cLocal = cQueue_.DeQue<float>();

            docQueue_.FreeTensor(docLocal);

            // ---- Vector：复用 v1 验证过的 ReduceMax Pattern::Reduce::AR ----
            LocalTensor<float> rowMax = rowMaxBuf_.Get<float>();
            uint32_t reduceShape2D[] = {mLen, kLen};
            ReduceMax<float, AscendC::Pattern::Reduce::AR>(rowMax, cLocal, reduceShape2D, false);
            AscendC::Max(runningMax, runningMax, rowMax, mLen);

            cQueue_.FreeTensor(cLocal);
        }

        // ---- 本核负责的 M 子块局部 ReduceSum ----
        LocalTensor<float> partialSum = partialSumBuf_.Get<float>();
        uint32_t reduceShape1D[] = {1, mLen};
        ReduceSum<float, AscendC::Pattern::Reduce::AR>(partialSum, runningMax, reduceShape1D, false);

        // ---- 跨核汇总：同一 batch 的不同 M 子块分布在不同核，v1 单核
        //      包办整个 batch 不需要这步，v2 引入多核切分后必须加 ----
        SetAtomicAdd<float>(true);
        DataCopy(scoreGm_[b], partialSum, 1);
        SetAtomicAdd<float>(false);
    }

    TPipe pipe_;
    ColbertMaxSimTilingV2 tiling_;
    AscendC::tiling::TCubeTiling cubeTiling_;

    GlobalTensor<half> qGm_, docGm_;
    GlobalTensor<float> scoreGm_;

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, half>,
           MatmulType<TPosition::VECIN, CubeFormat::ND, half>,
           MatmulType<TPosition::VECIN, CubeFormat::ND, float>>
        matmulObj_;

    TQue<QuePosition::VECIN, BUFFER_NUM> docQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> cQueue_;
    TBuf<QuePosition::VECCALC> rowMaxBuf_, runningMaxBuf_, partialSumBuf_;
};

extern "C" __global__ __aicore__ void colbert_maxsim_v2(GM_ADDR q, GM_ADDR doc, GM_ADDR score,
                                                         ColbertMaxSimTilingV2 tiling,
                                                         AscendC::tiling::TCubeTiling cubeTiling) {
    KernelColbertMaxSimV2 op;
    op.Init(q, doc, score, tiling, cubeTiling);
    op.Process();
}

// ================= host（新增，v1 没有多 batch 并发入口，这里补上）=================
int32_t main(int32_t argc, char **argv)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t physicalCoreNum = ascendcPlatform->GetCoreNumAiv();

    // 故意用一个"batch 数远小于核数"的场景，复现 v1 的问题：
    // v1 按 batch 发射 kernel，B=4 时 910B3 一堆核闲置；
    // v2 把 (B,M) 联合切分打散到全部核，才能真正用满硬件。
    constexpr uint32_t B = 4;
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 256;
    constexpr uint32_t D = 128;

    auto tiling = ComputeColbertMaxSimTilingV2(B, M, K, D, physicalCoreNum);
    auto cubeTiling = GenerateCubeTiling(ascendcPlatform, tiling.mTile, D, tiling.baseK);

    const size_t qSize = (size_t)B * M * D * sizeof(aclFloat16);
    const size_t dSize = (size_t)B * K * D * sizeof(aclFloat16);
    const size_t outSize = (size_t)B * sizeof(float);

    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *qDevice = nullptr, *dDevice = nullptr, *outDevice = nullptr;
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&qDevice), qSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&dDevice), dSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&outDevice), outSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(outDevice, outSize, 0, outSize));  // AtomicAdd 汇总前必须清零

    std::vector<aclFloat16> qHost(B * M * D);
    std::vector<aclFloat16> dHost(B * K * D);
    std::vector<float> outHost(B, 0.0f);
    std::vector<float> golden(B, 0.0f);

    srand(42);
    for (auto &v : qHost) v = aclFloatToFloat16((float)(rand() % 100) / 100.0f);
    for (auto &v : dHost) v = aclFloatToFloat16((float)(rand() % 100) / 100.0f);

    // golden: score[b] = sum_m max_k (Q[b,m,:] . Doc[b,k,:])
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t m = 0; m < M; m++) {
            float rowMax = -1e30f;
            for (uint32_t k = 0; k < K; k++) {
                float acc = 0.0f;
                for (uint32_t x = 0; x < D; x++) {
                    acc += aclFloat16ToFloat(qHost[b * M * D + m * D + x]) *
                           aclFloat16ToFloat(dHost[b * K * D + k * D + x]);
                }
                rowMax = std::max(rowMax, acc);
            }
            golden[b] += rowMax;
        }
    }

    CHECK_ACL(aclrtMemcpy(qDevice, qSize, qHost.data(), qSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dDevice, dSize, dHost.data(), dSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // 一次性发射 coreNum 个核，内部用 GetBlockIdx() 轮询消费全部
    // totalBlocks 个 (B,M) 任务块——和 v1 "每个 batch 单独发射一次" 不同，
    // 这是整份文件里体现"多核优化"的关键一行。
    colbert_maxsim_v2<<<tiling.coreNum, 0, stream>>>(qDevice, dDevice, outDevice, tiling, cubeTiling);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(outHost.data(), outSize, outDevice, outSize, ACL_MEMCPY_DEVICE_TO_HOST));

    std::printf("coreNum=%u totalBlocks=%u\n", tiling.coreNum, tiling.totalBlocks);
    std::printf("Output: ");
    for (uint32_t i = 0; i < B; i++) std::printf("%.2f ", outHost[i]);
    std::printf("\nGolden: ");
    for (uint32_t i = 0; i < B; i++) std::printf("%.2f ", golden[i]);
    std::printf("\n");

    bool pass = true;
    for (uint32_t i = 0; i < B; i++) {
        if (std::fabs(outHost[i] - golden[i]) > 0.05f) pass = false;
    }
    std::printf("test %s\n", pass ? "pass" : "failed");
    if (!pass) return 1;

    CHECK_ACL(aclrtFree(qDevice));
    CHECK_ACL(aclrtFree(dDevice));
    CHECK_ACL(aclrtFree(outDevice));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}
