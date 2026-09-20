// Host driver for simt_gm_memory.cce.
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <vector>

using namespace std;
static void check(int code, const char *name) {
  if (code) { fprintf(stderr, "%s: %d\n", name, code); exit(1); }
}
constexpr size_t GM_BYTES = 128ULL * 1024 * 1024;

static char *readBin(const char *f, uint32_t *sz) {
  ifstream s(f, ios::binary);
  if (!s) exit(1);
  s.seekg(0, ios::end);
  size_t n = s.tellg();
  if (!n) exit(1);
  s.seekg(0);
  char *b = new char[n];
  s.read(b, n);
  if (!s) exit(1);
  *sz = n;
  return b;
}

static void *reg(const char *bin, char **buf) {
  uint32_t sz;
  *buf = readBin(bin, &sz);
  rtDevBinary_t b = {};
  b.data = *buf;
  b.length = sz;
  b.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  b.version = 0;
  void *h = nullptr;
  check(rtDevBinaryRegister(&b, &h), "register binary");
  return h;
}

struct Args {
  void *out;
  void *gm;
  int K;
  int nlane;
  int nwarp;
  int iters;
  int mode;
};

static long long runK(const char *fn, rtStream_t stream, void *dout, void *gm,
                      int K, int nwarp, int iters, int mode) {
  Args args{dout, gm, K, 32, nwarp, iters, mode};
  rtArgsEx_t ai = {};
  ai.args = &args;
  ai.argsSize = sizeof(args);
  rtTaskCfgInfo_t cfg = {};
  cfg.localMemorySize = 192 * 1024;
  if (mode == 0) check(rtMemset(gm, GM_BYTES, 0, GM_BYTES), "initialize input");
  check(rtKernelLaunchWithFlagV2((void *)fn, 1, &ai, 0, stream, 0, &cfg), "launch");
  check(rtStreamSynchronize(stream), "synchronize");
  long long cycles = 0;
  check(rtMemcpy(&cycles, sizeof(cycles), dout, sizeof(cycles),
                 RT_MEMCPY_DEVICE_TO_HOST), "copy ticks");
  vector<float> values(nwarp * 32);
  check(rtMemcpy(values.data(), values.size() * sizeof(float),
                 static_cast<char *>(dout) + 32, values.size() * sizeof(float),
                 RT_MEMCPY_DEVICE_TO_HOST), "copy sink");
  for (int tid = 0; tid < nwarp * 32; ++tid)
    if (cycles <= 0 || values[tid] != tid + 1 + (mode ? iters : 0)) exit(2);
  if (mode) {
    vector<float> stores(static_cast<size_t>(iters) * nwarp * 32 * 8);
    check(rtMemcpy(stores.data(), stores.size() * sizeof(float), gm,
                   stores.size() * sizeof(float), RT_MEMCPY_DEVICE_TO_HOST), "copy stores");
    for (int i = 0; i < iters; ++i)
      for (int j = 0; j < 8; ++j)
        for (int tid = 0; tid < nwarp * 32; ++tid)
          if (stores[(i * 8 + j) * nwarp * 32 + tid] != tid + 1 + i + j) exit(2);
  }
  printf("sample,%d,%d,%d,%d,%lld,PASS\n", nwarp, mode, K, iters, cycles);
  return cycles;
}

static long long minimum(const char *fn, rtStream_t stream, void *dout,
                         void *gm, int K, int nwarp, int iters, int mode) {
  long long result = (long long)1e18;
  for (int rep = 0; rep < 5; ++rep) {
    long long value = runK(fn, stream, dout, gm, K, nwarp, iters, mode);
    if (value < result)
      result = value;
  }
  return result;
}

static double cyclesPerIter(const char *fn, rtStream_t stream, void *dout,
                            void *gm, int nwarp, int mode) {
  const int K = 4;
  const int I1 = 1024;
  const int I2 = 4096;
  long long c1 = minimum(fn, stream, dout, gm, K, nwarp, I1, mode);
  long long c2 = minimum(fn, stream, dout, gm, K, nwarp, I2, mode);
  return (double)(c2 - c1) / ((double)(I2 - I1) * K);
}

int main() {
  check(aclInit(nullptr), "init");
  check(rtSetDevice(0), "set device");
  char *binary = nullptr;
  void *handle = reg("simt_gm_memory.o", &binary);
  const char *fn = "measure";
  check(rtFunctionRegister(handle, fn, fn, (void *)fn, 0), "register function");
  rtStream_t stream;
  check(rtStreamCreate(&stream, 0), "stream");
  void *dout = nullptr;
  void *gm = nullptr;
  check(rtMalloc(&dout, 32 + 1024 * sizeof(float), RT_MEMORY_HBM, 0), "allocate output");
  check(rtMalloc(&gm, GM_BYTES, RT_MEMORY_HBM, 0), "allocate GM");
  check(rtMemset(gm, GM_BYTES, 0, GM_BYTES), "initialize GM");

  runK(fn, stream, dout, gm, 1, 4, 16, 0);
  printf("SIMT GM memory, 128 MiB maximum working set\n");
  printf("warps,load_cycles,load_bytes_per_cycle,store_cycles,"
         "store_bytes_per_cycle\n");
  for (int warps : {1, 2, 4, 8, 16, 32}) {
    double loadCycles = cyclesPerIter(fn, stream, dout, gm, warps, 0);
    double storeCycles = cyclesPerIter(fn, stream, dout, gm, warps, 1);
    double bytes = (double)warps * 32.0 * 8.0 * sizeof(float);
    printf("%d,%.6f,%.6f,%.6f,%.6f\n", warps, loadCycles, bytes / loadCycles,
           storeCycles, bytes / storeCycles);
  }

  check(rtFree(gm), "free GM");
  check(rtFree(dout), "free output");
  check(rtStreamDestroy(stream), "destroy stream");
  check(aclFinalize(), "finalize");
  delete[] binary;
  return 0;
}
