// Rút gọn từ https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/cuda/wkv_cuda.cu
#include <stdio.h>
#include <assert.h>
#include <torch/extension.h>

#define MIN_VALUE (-1e38)

// By using the __restrict__ keyword, the programmer informs the compiler that a particular pointer argument or variable 
// does not alias with any other pointer in the same scope. This allows the compiler to perform certain optimizations
// and generate more efficient code.

// Note: Đây là kernel của công thức GPT nhưng triển khai theo RNN chỉ để tránh overflow !!!
template <typename F>
__global__ void kernel_forward( const int B, const int T, const int C,
                                const F *__restrict__ const _w, 
                                const F *__restrict__ const _u,
                                const F *__restrict__ const _k,
                                const F *__restrict__ const _v,
                                F *__restrict__ const _y) {

    // Xác định index hiện tại trong mảng B * C
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // idx = _b * C + _c đại diện cho phần tử (_b, c) của ma trận (B, C)
    const int _b = idx / C; // xác định hàng _b: batch đang xét
    const int _c = idx % C; // xác định  cột _c: channel đang xét
    // Mỗi batch chứa _b phần tử (T * C) kiểu F (scalar)
    const int _offset = _b * T * C + _c; // offset để trỏ tới các scalar values của channel đang xét

    F u = _u[_c]; // u của channel đang xét
    F w = _w[_c]; // w của channel đang xét

    const F *__restrict__ const k = _k + _offset; // trỏ tới k của channel đang xét
    const F *__restrict__ const v = _v + _offset; // trỏ tới v của channel đang xét
    F *__restrict__ const y = _y + _offset; // trỏ tới giá trị đầu ra của channel đang xét
    
    F p = 0, q = 0, o = MIN_VALUE; // p and q are running sums divided by exp(o) (to avoid overflows)

    // Tính giá trị đầu ra bằng cách chạy dọc theo ctx_len T
    for (int i = 0; i < T; i++) {
        const int ii = i * C;

        F no = max(o, u + k[ii]);
        F A = exp(o - no);
        F B = exp(u + k[ii] - no);
        y[ii] = (A * p + B * v[ii]) / (A * q + B);

        no = max(w + o, k[ii]);
        A = exp(w + o - no);
        B = exp(k[ii] - no);
        p = A * p + B * v[ii];
        q = A * q + B;
        o = no;
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C,
                                const F *__restrict__ const _w,
                                const F *__restrict__ const _u,
                                const F *__restrict__ const _k,
                                const F *__restrict__ const _v,
                                const F *__restrict__ const _gy, // gradient đầu vào
                                F *__restrict__ const _gw, // gradient đầu ra của _w
                                F *__restrict__ const _gu, // gradient đầu ra của _u
                                F *__restrict__ const _gk, // gradient đầu ra của _k
                                F *__restrict__ const _gv) // gradient đầu ra của _v
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    F u = _u[_c];
    F w = _w[_c];

    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    const F *__restrict__ const gy = _gy + _offset;

    F *__restrict__ const gk = _gk + _offset;
    F *__restrict__ const gv = _gv + _offset;

    F y[Tmax], z[Tmax], zexp[Tmax];

    F gw = 0, gu = 0;
    F p = 0, q = 0;
    F dpdw = 0, dqdw = 0;
    F o = MIN_VALUE;

    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        F no = max(o, k[ii] + u);
        F A = exp(o - no);
        F B = exp(k[ii] + u - no);

        F num = A * p + B * v[ii];
        F iden = 1 / (A * q + B);

        y[i] = num * iden;
        z[i] = iden;
        zexp[i] = k[ii] + u - no;

        gw += gy[ii] * (dpdw - dqdw * y[i]) * iden * A;
        gu += gy[ii] * (v[ii] - y[i]) * B * iden;

        no = max(w + o, k[ii]);
        A = exp(w + o - no);
        B = exp(k[ii] - no);
        dpdw = A * (p + dpdw);
        dqdw = A * (q + dqdw);
        p = A * p + B * v[ii];
        q = A * q + B;
        o = no;
    }

    F gp = 0, gq = 0;
    o = MIN_VALUE;
    for (int i = T - 1; i >= 0; i--) {
        const int ii = i * C;
        F A = gy[ii] * z[i] * exp(zexp[i]);
        F B = exp(k[ii] + o);
        gk[ii] = A * (v[ii] - y[i]) + B * (gp * v[ii] + gq);
        gv[ii] = A + B * gp;

        F no = max(w + o, zexp[i] - k[ii] - u);
        A = exp(w + o - no);
        B = gy[ii] * z[i] * exp(zexp[i] - k[ii] - u - no);
        gp = A * gp + B;
        gq = A * gq - B * y[i];
        o = no;
    }

    // Multiply by w because the w -> -exp(w) preprocessing is halfway in the backwards pass,
    // even though it's not in the forward pass
    const int _offsetBC = _b * C + _c;
    _gw[_offsetBC] += gw * _w[_c];
    _gu[_offsetBC] += gu;
}


void forward(int64_t _B, int64_t _T, int64_t _C, 
        torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    const int B = _B, T = _T, C = _C; // convert i64 to i32

    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
 
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, 
        w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>());
}


void backward(int64_t _B, int64_t _T, int64_t _C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, 
        torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {    
    const int B = _B, T = _T, C = _C; // convert i64 to i32

    dim3 threadsPerBlock( min(C, 32) ); // lấy min đề phòng khi C < 32 dẫn tới trường hợp ko có thread nào đc launch
    assert(B * C % threadsPerBlock.x == 0); 
    dim3 numBlocks(B * C / threadsPerBlock.x);

    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, 
        w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), 
        gy.data_ptr<float>(), gw.data_ptr<float>(), gu.data_ptr<float>(), gk.data_ptr<float>(), gv.data_ptr<float>());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "cocktail_rwkv4_wkv_forward");
    m.def("backward", &backward, "cocktail_rwkv4_wkv_backward");
}

TORCH_LIBRARY(cocktail_rwkv4_wkv, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}