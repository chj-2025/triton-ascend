// Host driver for simt_memory.cce.
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace std;
static void check(int code, const char *op) {
  if (code) { fprintf(stderr, "%s: %d\n", op, code); exit(EXIT_FAILURE); }
}

static char *readBin(const char *f, uint32_t *sz) {
  ifstream s(f, ios::binary);
  if (!s) exit(EXIT_FAILURE);
  s.seekg(0, ios::end);
  size_t n = s.tellg();
  s.seekg(0);
  char *b = new char[n];
  s.read(b, n);
  *sz = n;
  return b;
}

static void *reg(const char *bin, char **buf) {
  uint32_t sz;
  *buf = readBin(bin, &sz);
  rtDevBinary_t b;
  b.data = *buf;
  b.length = sz;
  b.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  b.version = 0;
  void *h = nullptr;
  check(rtDevBinaryRegister(&b, &h), "rtDevBinaryRegister");
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
                      int nlane, int nwarp, int iters, int mode) {
  Args args{dout, K, nlane, nwarp, iters, mode};
  rtArgsEx_t ai = {};
  ai.args = &args;
  ai.argsSize = sizeof(args);
  rtTaskCfgInfo_t cfg = {};
  cfg.localMemorySize = 192 * 1024;
  check(rtKernelLaunchWithFlagV2((void *)fn, 1, &ai, 0, stream, 0, &cfg), "rtKernelLaunchWithFlagV2");
  check(rtStreamSynchronize(stream), "rtStreamSynchronize");
  long long ticks = 0;
  check(rtMemcpy(&ticks, sizeof(ticks), dout, sizeof(ticks),
                 RT_MEMCPY_DEVICE_TO_HOST), "copy results");
  vector<float> values(32768 + nwarp * 32);
  check(rtMemcpy(values.data(), values.size() * sizeof(float),
                 static_cast<char *>(dout) + 32, values.size() * sizeof(float),
                 RT_MEMCPY_DEVICE_TO_HOST), "copy UB");
  for (int tid = 0; tid < nwarp * 32; ++tid) {
    long long expected = tid + 1 + iters * (mode == 0 ? 8 : 1);
    float actual = values[32768 + tid];
    if (actual != expected) {
      fprintf(stderr, "Incorrect result tid=%d got=%f expected=%lld\n",
              tid, actual, expected);
      exit(EXIT_FAILURE);
    }
  }
  if (mode == 1) {
    for (int set = 0; set < 4; ++set) {
      int last = iters - 1 - ((iters - 1 - set) & 3);
      for (int j = 0; j < 8; ++j)
        for (int tid = 0; tid < nwarp * 32; ++tid)
          if (values[(set * 8 + j) * nwarp * 32 + tid] != tid + 1 + last + j)
            exit(EXIT_FAILURE);
    }
  }
  if (ticks <= 0) exit(EXIT_FAILURE);
  printf("RAW K=%d warps=%d iters=%d mode=%d ticks=%lld PASS\n",
         K, nwarp, iters, mode, ticks);
  return ticks;
}

static long long minimum(const char *fn, rtStream_t stream, void *dout, int K,
                         int nlane, int nwarp, int iters, int mode) {
  long long result = (long long)1e18;
  for (int rep = 0; rep < 7; ++rep) {
    long long value = runK(fn, stream, dout, K, nlane, nwarp, iters, mode);
    if (value < result)
      result = value;
  }
  return result;
}

static double cyclesPerIter(const char *fn, rtStream_t stream, void *dout,
                            int nwarp, int mode) {
  const int K = 20;
  const int I1 = 128;
  const int I2 = 512;
  long long c1 = minimum(fn, stream, dout, K, 32, nwarp, I1, mode);
  long long c2 = minimum(fn, stream, dout, K, 32, nwarp, I2, mode);
  return (double)(c2 - c1) / ((double)(I2 - I1) * K);
}

int main() {
  check(aclInit(nullptr), "aclInit");
  check(rtSetDevice(0), "rtSetDevice");
  char *binary = nullptr;
  void *handle = reg("simt_memory.o", &binary);
  const char *fn = "measure";
  check(rtFunctionRegister(handle, fn, fn, (void *)fn, 0), "rtFunctionRegister");
  rtStream_t stream;
  check(rtStreamCreate(&stream, 0), "rtStreamCreate");
  void *dout = nullptr;
  check(rtMalloc(&dout, 32 + (32768 + 1024) * sizeof(float), RT_MEMORY_HBM, 0), "rtMalloc");

  runK(fn, stream, dout, 2, 32, 4, 16, 0);
  printf("SIMT UB memory, eight operations/thread/iteration\n");
  printf("warps,load_cycles,load_bytes_per_cycle,store_cycles,"
         "store_bytes_per_cycle\n");
  for (int warps : {1, 2, 4, 8, 16, 32}) {
    double loadCycles = cyclesPerIter(fn, stream, dout, warps, 0);
    double storeCycles = cyclesPerIter(fn, stream, dout, warps, 1);
    double bytes = (double)warps * 32.0 * 8.0 * sizeof(float);
    printf("%d,%.6f,%.6f,%.6f,%.6f\n", warps, loadCycles, bytes / loadCycles,
           storeCycles, bytes / storeCycles);
  }

  check(rtFree(dout), "free");
  check(rtStreamDestroy(stream), "destroy stream");
  check(aclFinalize(), "finalize");
  delete[] binary;
  return 0;
}
