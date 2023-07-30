import torch, os, math
import torch.nn as nn
from torch.utils import cpp_extension


class RWKV_ChannelMix(nn.Module):

    def __init__(self, args, layer_id):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd): x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * args.n_embd
        self.key = nn.Linear(args.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, args.n_embd, bias=False)


    def forward(self, x):
        xx = self.time_shift(x) # do token mixing
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # square(relu(key @ xk)) can be fused
        k = self.key(xk)
        k = torch.relu(k)
        k = torch.square(k)

        # sigmoid(receptance @ xr) can be fused
        r = self.receptance(xr)
        r = torch.sigmoid(r)

        rkv = r * self.value(k) # kv
        return rkv


#########################################################################################


T_MAX = int(os.environ.get("RWKV_T_MAX", 2048)) # T_MAX chính là max của ctx_len, càng dài càng tốn vram

# Load nhân cuda
wkv_cuda = cpp_extension.load(
    name=f"cocktail_rwkv4_float32_wkv_{T_MAX}", 
    sources=["modules/rwkv4_timemix_wkv_cuda.cu"],
    verbose=True,
    extra_cuda_cflags=["-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60", "--use_fast_math", 
        "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"]
)


# Note: Chỉ hỗ trợ bf16 để loại bỏ if then
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B, ctx.T, ctx.C = B, T, C
        assert T <= T_MAX # Độ dài ctx_len phải <= T_MAX
        if C > 32: assert (B * C) % 32 == 0, "Nếu C > 32 thì B * C phải chia hết cho 32 để tối ưu cho nhân cuda"

        # biến thành f32 để tăng độ chính xác, 
        # và duỗi thành mảng 1 chiều để chuẩn bị feed cho nhân cuda
        w = -torch.exp(w.float().contiguous())
        u = u.float().contiguous() # giá trị khởi tạo t0
        k = k.float().contiguous() # k như trong trong KQV
        v = v.float().contiguous() # v như trong trong KQV

        # Chuẩn bị bộ nhớ các giá trị đầu ra
        y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y) # giá trị đầu ra được lưu vào y
        ctx.save_for_backward(w, u, k, v, y) # lưu lại giá trị để tính backward
        return y.half()


    @staticmethod
    def backward(ctx, gy):
        B, T, C = ctx.B, ctx.T, ctx.C
        w, u, k, v, y = ctx.saved_tensors

        gw = torch.zeros((B, C), device=gy.device).float().contiguous()
        gu = torch.zeros((B, C), device=gy.device).float().contiguous()
        gk = torch.zeros((B, T, C), device=gy.device).float().contiguous()
        gv = torch.zeros((B, T, C), device=gy.device).float().contiguous()

        gy_ = gy.float().contiguous() # biến đổi thành f32
        wkv_cuda.backward(B, T, C, w, u, k, v, gy_, gw, gu, gk, gv)

        del w; del u; del k; del v; del y # xóa saved tensors only!

        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)

        # Vì forward(ctx, B, T, C, w, u, k, v) nên backward cần trả lại từng đấy tham số (trừ ctx)
        # Đầu vào B, T, C không cần tính gradient nên giá trị trả về là None, None, None
        return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())


class RWKV_TimeMix(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        attn_sz = args.n_embd # chọn attention size bằng chiều của vector nhúng

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0

            # fancy time_decay
            decay_speed = [-5 + 8*(h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1) for h in range(attn_sz) ]
            self.time_decay = nn.Parameter(torch.tensor(decay_speed))
            # time_decay => -5.00, -3.16, -1.89, -0.78,  0.23,  1.20,  2.11,  3.00

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)

            # fancy time_mix
            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd): x[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))


        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1)) # padding zero trước embd vector đầu tiên trong batch
        self.key = nn.Linear(args.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(args.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(args.n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, args.n_embd, bias=False)


    def forward(self, x):
        xx = self.time_shift(x) # do token mixing
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)

        # sigmoid(receptance @ xr) can be fused
        r = self.receptance(xr)
        r = torch.sigmoid(r)

        B, T, C = x.size()
        rwkv = r * WKV.apply(B, T, C, self.time_decay, self.time_first, k, v)
        return self.output(rwkv)


######################################################################################


ZERO = ".att.key att.receptance .att.output .ffn.value .ffn.receptance .ffnPre.value \
    .ffnPre.receptance head_q. .oo. .rr.".strip().split()

def init_weight(self):
    m = {}
    for n, p in self.state_dict().items():

        if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n:
            if 'ln_x.weight' in n:
                layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                m[n] = (p * 0.0) + (layer_scale ** 0.5)
            else:
                m[n] = p

        else:
            shape = p.shape
            gain, scale = 1.0, 1.0
            if n == "emb.weight":
                scale = -1 * self.args.lr_init
            else:
                if n == "head.weight":
                    scale = 0.5
                else:
                    for kk in ZERO:
                        if kk in n: scale = 0; break

                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])

            print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(8)} {n}")
            x = torch.empty((shape[0], shape[1]))
            if scale == 0:  nn.init.zeros_(x)
            elif scale < 0: nn.init.uniform_(x, a=scale, b=-scale)
            else:           nn.init.orthogonal_(x, gain = gain*scale)

        m[n] = x.half()

    return m
