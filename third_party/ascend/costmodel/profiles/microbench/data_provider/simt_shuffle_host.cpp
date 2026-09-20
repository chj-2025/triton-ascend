// Host driver for simt_shuffle.cce.
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <vector>

using namespace std;

static void check(int code, const char *name) {
  if (code) {
    fprintf(stderr, "%s failed: %d\n", name, code);
    exit(1);
  }
}

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
  int K;
  int nlane;
  int nwarp;
  int iters;
  int mode;
};

static long long runK(const char *fn, rtStream_t stream, void *dout, int K,
                      int nwarp, int iters, int mode) {
  Args args{dout, K, 32, nwarp, iters, mode};
  rtArgsEx_t ai = {};
  ai.args = &args;
  ai.argsSize = sizeof(args);
  rtTaskCfgInfo_t cfg = {};
  cfg.localMemorySize = 192 * 1024;
  check(rtKernelLaunchWithFlagV2((void *)fn, 1, &ai, 0, stream, 0, &cfg), "launch");
  check(rtStreamSynchronize(stream), "synchronize");
  long long cycles = 0;
  check(rtMemcpy(&cycles, sizeof(cycles), dout, sizeof(cycles),
                 RT_MEMCPY_DEVICE_TO_HOST), "copy ticks");
  vector<float> values(nwarp * 32);
  check(rtMemcpy(values.data(), values.size() * sizeof(float),
                 static_cast<char *>(dout) + 32, values.size() * sizeof(float),
                 RT_MEMCPY_DEVICE_TO_HOST), "copy output");
  for (int warp = 0; warp < nwarp; ++warp) {
    float x[4][32];
    for (int chain = 0; chain < 4; ++chain)
      for (int lane = 0; lane < 32; ++lane)
        x[chain][lane] = lane + warp + 1 + chain;
    for (int i = 0; i < iters; ++i)
      for (int chain = 0; chain < (mode ? 4 : 1); ++chain)
        for (int lane = 31; lane >= (1 << chain); --lane)
          x[chain][lane] = x[chain][lane - (1 << chain)];
    for (int lane = 0; lane < 32; ++lane) {
      float expected = x[0][lane] + x[1][lane] + x[2][lane] + x[3][lane];
      if (values[warp * 32 + lane] != expected || cycles <= 0) {
        fprintf(stderr, "shuffle mismatch: warp=%d lane=%d\n", warp, lane);
        exit(2);
      }
    }
  }
  printf("sample,%d,%d,%d,%d,%lld,PASS\n", nwarp, mode, K, iters, cycles);
  return cycles;
}

static long long minimum(const char *fn, rtStream_t stream, void *dout, int K,
                         int nwarp, int iters, int mode) {
  long long result = (long long)1e18;
  for (int rep = 0; rep < 7; ++rep) {
    long long value = runK(fn, stream, dout, K, nwarp, iters, mode);
    if (value < result)
      result = value;
  }
  return result;
}

static double cyclesPerIter(const char *fn, rtStream_t stream, void *dout,
                            int nwarp, int mode) {
  const int K = 20;
  const int I1 = 256;
  const int I2 = 1024;
  long long c1 = minimum(fn, stream, dout, K, nwarp, I1, mode);
  long long c2 = minimum(fn, stream, dout, K, nwarp, I2, mode);
  return (double)(c2 - c1) / ((double)(I2 - I1) * K);
}

int main() {
  check(aclInit(nullptr), "aclInit");
  check(rtSetDevice(0), "set device");
  char *binary = nullptr;
  void *handle = reg("simt_shuffle.o", &binary);
  const char *fn = "measure";
  check(rtFunctionRegister(handle, fn, fn, (void *)fn, 0), "register function");
  rtStream_t stream;
  check(rtStreamCreate(&stream, 0), "stream");
  void *dout = nullptr;
  check(rtMalloc(&dout, 32 + 1024 * sizeof(float), RT_MEMORY_HBM, 0), "allocate");

  runK(fn, stream, dout, 2, 4, 16, 0);
  printf("SIMT __shfl_up dependent and ILP4 throughput\n");
  printf("warps,dependent_cycles,dependent_warp_shuffles_per_cycle,"
         "ilp4_cycles,ilp4_warp_shuffles_per_cycle\n");
  for (int warps : {1, 2, 4, 8, 16, 32}) {
    double depCycles = cyclesPerIter(fn, stream, dout, warps, 0);
    double ilpCycles = cyclesPerIter(fn, stream, dout, warps, 1);
    printf("%d,%.6f,%.6f,%.6f,%.6f\n", warps, depCycles,
           (double)warps / depCycles, ilpCycles,
           (double)warps * 4.0 / ilpCycles);
  }

  check(rtFree(dout), "free");
  check(rtStreamDestroy(stream), "destroy stream");
  check(aclFinalize(), "finalize");
  delete[] binary;
  return 0;
}
