// fb.cu: the real low-precision path on one Blackwell GPU, callable from PyTorch through ctypes.
//   - cuBLASLt GEMMs on raw device pointers: BF16, FP8 (E4M3, one scale per tensor), NVFP4 (E2M1 + UE4M3 block scales)
//   - our own kernels that turn BF16/FP32 tensors into packed NVFP4 or FP8 bytes, with optional GELU and stochastic rounding
// Build: nvcc -O3 -std=c++17 -arch=sm_100a -shared -Xcompiler -fPIC fb.cu -o libfb.so -lcublasLt
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

// ------------------------------------------------------------------------------------------------ cuBLASLt wrapper
static cublasLtHandle_t H = nullptr;
struct Plan {
  cublasLtMatmulDesc_t desc; cublasLtMatrixLayout_t LA, LB, LC; cublasLtMatmulHeuristicResult_t hr[8];
  int n, pick, kind;
};

extern "C" int fb_init() { if (H) return 0; return (int)cublasLtCreate(&H); }

// kind: 0 BF16, 1 FP8 E4M3 with scalar scales, 2 NVFP4. C (c_rows x c_cols, column-major) = alpha * op(A) * op(B).
// Rows and columns are those of the stored, column-major matrices. out_type: 0 BF16, 1 FP32.
extern "C" void* fb_plan(int kind, int transA, int transB, long a_rows, long a_cols, long b_rows, long b_cols,
                         long c_rows, long c_cols, int out_type, size_t ws, const void* sA, const void* sB, int* status) {
  Plan* p = new Plan(); p->kind = kind; p->pick = -1; p->n = 0;
  cudaDataType ta = kind == 0 ? CUDA_R_16BF : (kind == 1 ? CUDA_R_8F_E4M3 : CUDA_R_4F_E2M1);
  cudaDataType tc = out_type == 0 ? CUDA_R_16BF : CUDA_R_32F;
  cublasStatus_t s;
#define TRY(x, code) do { s = (x); if (s != CUBLAS_STATUS_SUCCESS) { *status = (code) * 1000 + (int)s; return nullptr; } } while (0)
  TRY(cublasLtMatmulDescCreate(&p->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), 1);
  cublasOperation_t oa = transA ? CUBLAS_OP_T : CUBLAS_OP_N, ob = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  TRY(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof(oa)), 2);
  TRY(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_TRANSB, &ob, sizeof(ob)), 3);
  if (kind == 2) {
    cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    TRY(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)), 4);
    TRY(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)), 5);
  }
  if (kind >= 1) {
    TRY(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sA, sizeof(sA)), 6);
    TRY(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sB, sizeof(sB)), 7);
  }
  TRY(cublasLtMatrixLayoutCreate(&p->LA, ta, a_rows, a_cols, a_rows), 8);
  TRY(cublasLtMatrixLayoutCreate(&p->LB, ta, b_rows, b_cols, b_rows), 9);
  TRY(cublasLtMatrixLayoutCreate(&p->LC, tc, c_rows, c_cols, c_rows), 10);
  cublasLtMatmulPreference_t pref;
  TRY(cublasLtMatmulPreferenceCreate(&pref), 11);
  TRY(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)), 12);
  TRY(cublasLtMatmulAlgoGetHeuristic(H, p->desc, p->LA, p->LB, p->LC, p->LC, pref, 8, p->hr, &p->n), 13);
  cublasLtMatmulPreferenceDestroy(pref);
  for (int i = 0; i < p->n; i++) if (p->hr[i].state == CUBLAS_STATUS_SUCCESS) { p->pick = i; break; }
  *status = p->pick < 0 ? 999000 : 0;
  return p;
}
extern "C" int fb_nalgo(void* pp) { return ((Plan*)pp)->n; }
extern "C" int fb_pick(void* pp, int i) { Plan* p = (Plan*)pp; if (i < 0 || i >= p->n || p->hr[i].state != CUBLAS_STATUS_SUCCESS) return 1; p->pick = i; return 0; }

extern "C" int fb_run(void* pp, float alpha, const void* A, const void* B, void* C, const void* sA, const void* sB,
                      void* ws, size_t wsz) {
  Plan* p = (Plan*)pp;
  if (p->kind >= 1) {
    cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sA, sizeof(sA));
    cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sB, sizeof(sB));
  }
  float beta = 0.0f;
  return (int)cublasLtMatmul(H, p->desc, &alpha, A, p->LA, B, p->LB, &beta, C, p->LC, C, p->LC, &p->hr[p->pick].algo, ws, wsz, 0);
}

// ------------------------------------------------------------------------------------------------ device helpers
__device__ __forceinline__ float bf16_to_f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }
__device__ __forceinline__ uint16_t f_to_bf16(float f) { uint32_t b = __float_as_uint(f); b += 0x7FFFu + ((b >> 16) & 1u); return (uint16_t)(b >> 16); }
__device__ __forceinline__ uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
__device__ __forceinline__ float urand(uint32_t key) { return (float)(hash32(key) >> 8) * (1.0f / 16777216.0f); }
__device__ __forceinline__ float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678f)); }
__device__ __forceinline__ float gelu_grad(float x) { return 0.5f * (1.0f + erff(x * 0.70710678f)) + x * 0.3989422804f * __expf(-0.5f * x * x); }

// UE4M3 block scale: sizes 2^-9 .. 448, never zero
__device__ __forceinline__ float dec_ue4m3(uint8_t c) {
  int ef = (c >> 3) & 15, m = c & 7;
  return ef == 0 ? (float)m * 0.001953125f : __uint_as_float(((uint32_t)(ef - 7 + 127) << 23) | ((uint32_t)m << 20));
}
// E4M3 magnitude code (7 bits) of a >= 0, rounded to nearest (sr = 0) or stochastically (u in [0,1))
__device__ __forceinline__ uint8_t enc_e4m3_mag(float a, int sr, float u) {
  int e = (int)(__float_as_uint(a) >> 23) - 127;
  e = min(max(e, -6), 8);
  float inv = __uint_as_float((uint32_t)(3 - e + 127) << 23);          // 1 / step, step = 2^(e-3)
  float t = a * inv;
  float q = sr ? floorf(t + u) : rintf(t);
  float val = fminf(q * __uint_as_float((uint32_t)(e - 3 + 127) << 23), 448.0f);
  uint32_t vb = __float_as_uint(val);
  if (val < 0.015625f) return (uint8_t)(val * 512.0f);
  return (uint8_t)((((int)(vb >> 23) - 127 + 7) << 3) | ((vb >> 20) & 7u));
}
// E2M1: bit 3 sign, bits 2..0 index the sizes 0, .5, 1, 1.5, 2, 3, 4, 6
__device__ __forceinline__ uint32_t enc_e2m1(float t) {
  float a = fabsf(t);
  float step = a < 2.0f ? 0.5f : (a < 4.0f ? 1.0f : 2.0f);
  float v = fminf(rintf(a / step) * step, 6.0f);
  uint32_t idx = v < 2.0f ? (uint32_t)(v * 2.0f) : (v < 4.0f ? 2u + (uint32_t)v : 4u + (uint32_t)(v * 0.5f));
  return idx | (t < 0.0f ? 8u : 0u);
}
__device__ __forceinline__ float dec_e2m1_mag(uint32_t c) {
  const float tab[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return tab[c & 7];
}

// ------------------------------------------------------------------------------------------------ kernels
// One thread per block of 16. Input rows x K (BF16 or FP32), optional GELU first. Output: 16 codes in one 64-bit word,
// the block scale written straight into cuBLASLt's scale layout, and optionally an FP8 copy of the QUANTIZED values
// (stochastically rounded), which is what the backward GEMMs read.
template <int IS_BF16>
__global__ void k_quant_nvfp4(const void* xin, long rows, int nb, float outer, uint64_t* codes, uint8_t* scales,
                              uint8_t* f8, float inv_s8, int do_gelu, uint32_t seed, int copy_sr) {
  long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= rows * nb) return;
  long r = b / nb; int kb = (int)(b - r * nb);
  float v[16], amax = 0.0f;
  if (IS_BF16) { const uint16_t* p = (const uint16_t*)xin + b * 16; for (int i = 0; i < 16; ++i) v[i] = bf16_to_f(p[i]); }
  else { const float* p = (const float*)xin + b * 16; for (int i = 0; i < 16; ++i) v[i] = p[i]; }
  if (do_gelu) for (int i = 0; i < 16; ++i) v[i] = gelu(v[i]);
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
  uint8_t sc = max((int)enc_e4m3_mag(amax / 6.0f / outer, 0, 0.0f), 1);
  float scale = dec_ue4m3(sc) * outer;
  uint64_t w = 0;
  uint32_t key = seed ^ hash32((uint32_t)b * 16u + 0x9e3779b9U);
  for (int i = 0; i < 16; ++i) {
    uint32_t c = enc_e2m1(v[i] / scale);
    w |= (uint64_t)c << (4 * i);
    if (f8) { float xq = dec_e2m1_mag(c) * scale; f8[b * 16 + i] = enc_e4m3_mag(xq * inv_s8, copy_sr, urand(key + (uint32_t)i)) | ((c & 8u) ? 0x80 : 0); }
  }
  codes[b] = w;
  long off = ((r >> 7) * (nb >> 2) + (kb >> 2)) * 512 + (r & 31) * 16 + ((r & 127) >> 5) * 4 + (kb & 3);
  scales[off] = sc;
}

// One thread per 16 values: BF16 -> FP8 E4M3 with one tensor scale. Options: GELU first; multiply by GELU'(y) of a
// second tensor (the backward pass through the activation); nearest or stochastic rounding.
__global__ void k_quant_fp8(const uint16_t* xin, const uint16_t* y_for_grad, long n, float inv_s, uint8_t* out, int sr,
                            int do_gelu, uint32_t seed) {
  long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long base = b * 16;
  if (base >= n) return;
  uint32_t key = seed ^ hash32((uint32_t)b * 16u + 0x51ed270bU);
  for (int i = 0; i < 16 && base + i < n; ++i) {
    float v = bf16_to_f(xin[base + i]);
    if (do_gelu) v = gelu(v);
    if (y_for_grad) v *= gelu_grad(bf16_to_f(y_for_grad[base + i]));
    out[base + i] = enc_e4m3_mag(fabsf(v) * inv_s, sr, sr ? urand(key + (uint32_t)i) : 0.0f) | (v < 0.0f ? 0x80 : 0);
  }
}
template <int IS_BF16>
__global__ void k_quant_fp8_any(const void* xin, long n, float inv_s, uint8_t* out, int sr, uint32_t seed) {
  long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long base = b * 16;
  if (base >= n) return;
  uint32_t key = seed ^ hash32((uint32_t)b * 16u + 0x51ed270bU);
  for (int i = 0; i < 16 && base + i < n; ++i) {
    float v = IS_BF16 ? bf16_to_f(((const uint16_t*)xin)[base + i]) : ((const float*)xin)[base + i];
    out[base + i] = enc_e4m3_mag(fabsf(v) * inv_s, sr, sr ? urand(key + (uint32_t)i) : 0.0f) | (v < 0.0f ? 0x80 : 0);
  }
}


// ------------------------------------------------------------------------------------------------ kernels, version 2
// The same two jobs, with the rounding and encoding done by Blackwell's own conversion instructions
// (cvt ... e2m1x2 / e4m3x2: two values per instruction) instead of compares, shifts and masks.
// Stochastic rounding = add a uniform dither of one grid step, then let the hardware round to nearest.
// The clamp to the bottom of the binade keeps it exact where the grid spacing changes.
__device__ __forceinline__ float dec_e2m1_bits(uint32_t c) {
  uint32_t e = (c >> 1) & 3u, m = c & 1u;
  return e == 0 ? 0.5f * (float)m : __uint_as_float(((e - 1u + 127u) << 23) | (m << 22));
}
__device__ __forceinline__ float sr_dither(float a, float u) {                 // a >= 0, in units of the tensor scale; u in [0, 1)
  int e = (int)(__float_as_uint(a) >> 23) - 127;
  e = min(max(e, -6), 8);
  float step = __uint_as_float((uint32_t)(e - 3 + 127) << 23);
  float lo = a >= 0.015625f ? __uint_as_float((uint32_t)(e + 127) << 23) : 0.0f;
  return fmaxf(a + (u - 0.5f) * step, lo);
}
__device__ __forceinline__ void load16(const void* xin, int is_bf16, long b, float* v) {
  if (is_bf16) {
    const uint64_t* p = (const uint64_t*)xin + b * 4;
    for (int j = 0; j < 4; ++j) { uint64_t w = p[j]; for (int i = 0; i < 4; ++i) v[4 * j + i] = __uint_as_float(((uint32_t)(w >> (16 * i)) & 0xFFFFu) << 16); }
  } else { const float4* p = (const float4*)xin + b * 4; for (int j = 0; j < 4; ++j) { float4 q = p[j]; v[4 * j] = q.x; v[4 * j + 1] = q.y; v[4 * j + 2] = q.z; v[4 * j + 3] = q.w; } }
}

__global__ void k2_quant_nvfp4(const void* xin, int is_bf16, long rows, int nb, float outer, uint64_t* codes, uint8_t* scales,
                               uint64_t* f8, float inv_s8, int do_gelu, uint32_t seed, int copy_sr) {
  long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= rows * nb) return;
  long r = b / nb; int kb = (int)(b - r * nb);
  float v[16], amax = 0.0f;
  load16(xin, is_bf16, b, v);
  if (do_gelu) for (int i = 0; i < 16; ++i) v[i] = gelu(v[i]);
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
  uint8_t sc = max((int)enc_e4m3_mag(amax / 6.0f / outer, 0, 0.0f), 1);
  float scale = dec_ue4m3(sc) * outer, inv = 1.0f / scale;
  uint64_t w = 0, o0 = 0, o1 = 0;
  uint32_t key = seed ^ hash32((uint32_t)b * 16u + 0x9e3779b9U);
  float k8 = scale * inv_s8;
  for (int j = 0; j < 8; ++j) {
    uint32_t by = (uint32_t)__nv_cvt_float2_to_fp4x2(make_float2(v[2 * j] * inv, v[2 * j + 1] * inv), __NV_E2M1, cudaRoundNearest);
    w |= (uint64_t)by << (8 * j);
    if (f8) {
      uint32_t h = hash32(key + (uint32_t)j);
      float u0 = copy_sr ? (float)(h & 0xFFFFu) * (1.0f / 65536.0f) : 0.5f, u1 = copy_sr ? (float)(h >> 16) * (1.0f / 65536.0f) : 0.5f;   // 0.5 = no dither = to nearest
      float a0 = sr_dither(dec_e2m1_bits(by) * k8, u0), a1 = sr_dither(dec_e2m1_bits(by >> 4) * k8, u1);
      uint32_t s2 = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2((by & 8u) ? -a0 : a0, (by & 128u) ? -a1 : a1), __NV_SATFINITE, __NV_E4M3);
      if (j < 4) o0 |= (uint64_t)s2 << (16 * j); else o1 |= (uint64_t)s2 << (16 * (j - 4));
    }
  }
  codes[b] = w;
  if (f8) { f8[2 * b] = o0; f8[2 * b + 1] = o1; }
  long off = ((r >> 7) * (nb >> 2) + (kb >> 2)) * 512 + (r & 31) * 16 + ((r & 127) >> 5) * 4 + (kb & 3);
  scales[off] = sc;
}

__global__ void k2_quant_fp8(const void* xin, int is_bf16, const void* y_for_grad, long nblk, float inv_s, uint64_t* out, int sr,
                             int do_gelu, uint32_t seed) {
  long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblk) return;
  float v[16];
  load16(xin, is_bf16, b, v);
  if (do_gelu) for (int i = 0; i < 16; ++i) v[i] = gelu(v[i]);
  if (y_for_grad) { float y[16]; load16(y_for_grad, 1, b, y); for (int i = 0; i < 16; ++i) v[i] *= gelu_grad(y[i]); }
  uint32_t key = seed ^ hash32((uint32_t)b * 16u + 0x51ed270bU);
  uint64_t o0 = 0, o1 = 0;
  for (int j = 0; j < 8; ++j) {
    float a0 = fabsf(v[2 * j]) * inv_s, a1 = fabsf(v[2 * j + 1]) * inv_s;
    if (sr) { uint32_t h = hash32(key + (uint32_t)j); a0 = sr_dither(a0, (float)(h & 0xFFFFu) * (1.0f / 65536.0f)); a1 = sr_dither(a1, (float)(h >> 16) * (1.0f / 65536.0f)); }
    uint32_t s2 = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(copysignf(a0, v[2 * j]), copysignf(a1, v[2 * j + 1])), __NV_SATFINITE, __NV_E4M3);
    if (j < 4) o0 |= (uint64_t)s2 << (16 * j); else o1 |= (uint64_t)s2 << (16 * (j - 4));
  }
  out[2 * b] = o0; out[2 * b + 1] = o1;
}

static const int TH = 256;
static int COPY_SR = 1;                              // stochastic (1) or nearest (0) rounding for the 8-bit copies
extern "C" void fb_set_copy_sr(int v) { COPY_SR = v; }
static int HW = 1;                                   // 1: hardware conversion instructions, 0: the software kernels above
extern "C" void fb_set_hw(int hw) { HW = hw; }
extern "C" void fb_quant_nvfp4(const void* x, int is_bf16, long rows, long K, float outer, void* codes, void* scales,
                               void* f8, float s8, int do_gelu, unsigned seed) {
  int nb = (int)(K / 16); long n = rows * nb;
  unsigned g = (unsigned)((n + TH - 1) / TH);
  if (HW) k2_quant_nvfp4<<<g, TH>>>(x, is_bf16, rows, nb, outer, (uint64_t*)codes, (uint8_t*)scales, (uint64_t*)f8, 1.0f / s8, do_gelu, seed, COPY_SR);
  else if (is_bf16) k_quant_nvfp4<1><<<g, TH>>>(x, rows, nb, outer, (uint64_t*)codes, (uint8_t*)scales, (uint8_t*)f8, 1.0f / s8, do_gelu, seed, COPY_SR);
  else k_quant_nvfp4<0><<<g, TH>>>(x, rows, nb, outer, (uint64_t*)codes, (uint8_t*)scales, (uint8_t*)f8, 1.0f / s8, do_gelu, seed, COPY_SR);
}
// n must be a multiple of 16
extern "C" void fb_quant_fp8(const void* x, const void* y_for_grad, long n, float s, void* out, int sr, int do_gelu, unsigned seed) {
  unsigned g = (unsigned)((n / 16 + TH) / TH);
  if (HW) k2_quant_fp8<<<g, TH>>>(x, 1, y_for_grad, n / 16, 1.0f / s, (uint64_t*)out, sr, do_gelu, seed);
  else k_quant_fp8<<<g, TH>>>((const uint16_t*)x, (const uint16_t*)y_for_grad, n, 1.0f / s, (uint8_t*)out, sr, do_gelu, seed);
}
extern "C" void fb_quant_fp8_f32(const void* x, long n, float s, void* out, int sr, unsigned seed) {
  unsigned g = (unsigned)((n / 16 + TH) / TH);
  if (HW) k2_quant_fp8<<<g, TH>>>(x, 0, nullptr, n / 16, 1.0f / s, (uint64_t*)out, sr, 0, seed);
  else k_quant_fp8_any<0><<<g, TH>>>(x, n, 1.0f / s, (uint8_t*)out, sr, seed);
}
extern "C" int fb_last_error() { return (int)cudaGetLastError(); }
