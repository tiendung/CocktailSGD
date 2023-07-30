import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import random, types, torch
from torch.nn import functional as F

args = types.SimpleNamespace()
args.MODEL_NAME = "./model_ckpts/rwkv4-150m-cocktail/checkpoint_300/prank_0_checkpoint.pt"
args.tokenizer_name = "./empty_model_configs/rwkv4-150m"
args.n_layer = 12
args.n_embd = 768

context = "\nOnce upon a time, there was a little boy named Tim. Tim loved to collect things."

from modules.tokenizer import build_tokenizer
tokenizer = build_tokenizer(args)

NUM_TRIALS = 3
LENGTH_PER_TRIAL = 250
TEMPERATURE = 0.6
TOP_P = 0.85


########################################################################################################
# Step 1: Define the model in inference mode
########################################################################################################

class RWKV_RNN(torch.jit.ScriptModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eval() # set torch to inference mode

        # Load tham số từ file vào vào bộ nhớ và biến đổi cho phù hợp
        w = torch.load(args.MODEL_NAME, map_location='cpu')

        for k in w.keys():
            if      '.time_' in k: w[k] = w[k].squeeze() # (A,1,B,1) => (A,B)
            if '.time_decay' in k: w[k] = -torch.exp(w[k].float())
            # => e^negative = decay it's actually `e^{-e^x}`
            else: w[k] = w[k].float() # convert to f32 type

        ''' (( GÁN THAM SỐ VÀO BIẾN SỐ MÔ HÌNH ))
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -       
        Tên tham số (string)                     Biến số (namespace)
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -       
        "0.word_embeddings.weight"            => self.w.weight
        '0.word_embeddings_layernorm.weight'  => self.w.blocks[0].ln0.weight
        '0.word_embeddings_layernorm.bias'    => self.w.blocks[0].ln0.bias
        "17.ln_f.weight"                      => self.w.ln_out.weight
        "17.ln_f.bias"                        => self.w.ln_out.bias
        "17.lm_head.weight"                   => self.w.head.weight
        "1.ln1.weight"                        => self.w.blocks[0].ln1.weight
        "1.ln1.bias"                          => self.w.blocks[0].ln1.bias
        "1.ln2.weight"                        => self.w.blocks[0].ln2.weight
        "1.ln2.bias"                          => self.w.blocks[0].ln2.bias
        "1.att.time_first"                    => self.w.blocks[0].att.time_first        
        "1.att.time_decay"                    ...
        "1.att.time_mix_k"
        "1.att.time_mix_v"
        "1.att.time_mix_r"
        "1.att.key.weight"
        "1.att.value.weight"
        "1.att.receptance.weight"
        "1.att.output.weight"
        "1.ffn.time_mix_k"
        "1.ffn.time_mix_r"
        "1.ffn.key.weight"
        "1.ffn.value.weight"
        "1.ffn.receptance.weight"
                                      ... tiếp tục cho tới block thứ n ...
        '''
        self.w = types.SimpleNamespace()
        self.w.blocks = {}

        for k in w.keys():
            parts = k.split('.')
            parts[0] = int(parts[0])
            # bỏ qua tầng đầu và tầng cuối, gán thẳng giá trị ở bên dưới
            if parts[0] == 0 or parts[0] == args.n_layer + 1: continue

            print(">>>", parts)# ['blocks','0','ln0','weight']
            last = parts.pop() # => last = "weight"; parts = ['blocks','0','ln0']       
            here = self.w.blocks

            for p in parts: # từng bước mở rộng namespace
                if isinstance(p, int): # tầng thứ p
                    p -= 1 # dịch chuyển 1 => 0, ...
                    if p not in here:
                        # dùng [] vì here (w.blocks) là dict object {}
                        here[p] = types.SimpleNamespace()
                    here = here[p]

                else: # dùng hasattr, setattr, getattr vì here là types.SimpleNamespace()
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())  
                    here = getattr(here, p)

            setattr(here, last, w[k]) # gán giá trị vào namespace cuối cùng
            # => self.w.blocks[0].ln0.weight = w[k]

        self.w.emb                  = types.SimpleNamespace()
        self.w.emb.weight           = w["0.word_embeddings.weight"]

        self.w.blocks[0].ln0        = types.SimpleNamespace()
        self.w.blocks[0].ln0.weight = w['0.word_embeddings_layernorm.weight']
        self.w.blocks[0].ln0.bias   = w['0.word_embeddings_layernorm.bias']

        self.w.ln_out        = types.SimpleNamespace()
        self.w.ln_out.weight = w[f"{args.n_layer + 1}.ln_f.weight"]
        self.w.ln_out.bias   = w[f"{args.n_layer + 1}.ln_f.bias"]

        self.w.head          = types.SimpleNamespace()
        self.w.head.weight   = w[f"{args.n_layer + 1}.lm_head.weight"]


    """
    state[] để lưu trạng thái của rnn, mỗi tầng i ghi lại 5 trạng thái: 
    i+0 = ffn_xx : token của bước channel-mixing trước 
    i+1 = att_xx : token của bước time-mixing trước
    i+2 = att_aa : exp moving avg của kv 
    i+3 = att_bb : exp moving avg của k
    i+4 = att_pp : use pp to stove exponent of aa and bb
    """
    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    @torch.jit.script_method
    def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        ''' channel-mixing giống FFN của transformer nhưng có thêm nhiều cải tiến:
        * Token-shift trộn vector hiện tại với vector đầu vào trước đó theo 1 tỉ lệ nhất định, mục đích là để
          mô hình mang thông tin từ quá khứ tới hiện tại tốt hơn, giúp việc suy diễn tốt hơn
        * Dùng relu square (giống primer)
        * Nhân với hàm sigmoid(r) giúp mô hình ổn định hơn.
        '''
        ffn_xx = 5*i+0 # feed-forward or channel mixing
        # token-shift with diff mixing factors for k and r
        xk = x * time_mix_k + state[ffn_xx] * (1 - time_mix_k)
        xr = x * time_mix_r + state[ffn_xx] * (1 - time_mix_r)
        state[ffn_xx] = x # prev_x = x

        r = torch.sigmoid(rw @ xr) # receptance factor thuộc 0 -> 1
        k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
        return r * (vw @ k)

    @torch.jit.script_method
    def time_mixing(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        ''' Time-mixing hay còn gọi là linear-attention. Tuy nhiên công thức linear-attention này có thể được viết lại
        dưới dạng hồi quy nên sử dụng dạng công thức hồi quy để tính toán tiết kiệm hơn. Do hồi quy chỉ cần trạng thái
        hệ thống ở t-1 để tính ra trạng trái hệ thống ở t, không cần phải tính lại cho toàn bộ chuỗi đầu vào.
        '''
        att_xx = 5*i+1 # attention or time mixing
        # token-shift with diff mixing factors for k, v and r
        xk = x * time_mix_k + state[att_xx] * (1 - time_mix_k)
        xv = x * time_mix_v + state[att_xx] * (1 - time_mix_v)
        xr = x * time_mix_r + state[att_xx] * (1 - time_mix_r)
        state[att_xx] = x # prev_x = x

        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv

        # Công thức hồi quy của rnn mode, xem https://github.com/telexyz/symato/blob/main/docs/rwkv-illustrated.md
        aa = state[5*i+2] # exponential moving average of kv
        bb = state[5*i+3] # exponential moving average of k
        pp = state[5*i+4] # idea: use pp to store exponent of a and b

        ww = time_first + k # u + k_i
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b

        ww = pp + time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = qq

        return ow @ (r * wkv)

    def forward(self, token_id, state, preprocess_only=False):
        with torch.no_grad():
            # 0/ Khởi tạo trạng thái hệ thống nếu chưa được khởi tạo
            if state == None:
                state = torch.zeros(self.args.n_layer * 5, self.args.n_embd)
                for i in range(self.args.n_layer): state[5*i+4] = -1e30 # state[att_pp] = âm vô cực

            # 1/ Lấy vector nhúng của token_id
            x = self.w.emb.weight[token_id]
            # Và áp dụng layer-norm-0 ở tầng đầu tiên để small-init-emb trick hoạt động
            x = self.layer_norm(x, self.w.blocks[0].ln0)
            
            # 2/ Với mỗi tầng áp dụng:
            for i in range(self.args.n_layer):
                # 2.1/ time-mixing
                att = self.w.blocks[i].att # trọng số của khối time-mixing
                x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i, 
                    att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay, 
                    att.key.weight, att.value.weight, att.receptance.weight, att.output.weight)

                # 2.2/ channel-mixing
                ffn = self.w.blocks[i].ffn # trọng số của khối channel-mixing
                x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i, 
                    ffn.time_mix_k, ffn.time_mix_r, 
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)

            if preprocess_only: return state
            # 3/ Cuối cùng áp dụng bộ phân lớp cho ra next token probabilities
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state

########################################################################################################
# Step 2: set prompt & sampling stuffs
########################################################################################################

def sample_logits(out, temperature=0.2, top_p_usual=0.8):
    probs = F.softmax(out, dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs) # [1,2,3] => [1,3,6]
    idx = np.argmax(cumulative_probs > top_p_usual) # vì là mảng True, False nên trả về idx của True đầu tiên
    cutoff = float(sorted_probs[idx]) # cutoff là tổng những prob lớn nhất đầu tiên vượt qua top_p_usual
    probs[probs < cutoff] = 0 # bỏ đi những prob < cutoff
    if temperature != 1.0: probs = np.power(probs, 1.0 / temperature)
    probs = probs / np.sum(probs) # chuẩn hóa lại probs sao cho tổng = 1
    return np.random.choice(a=len(probs), p=probs) # lấy mẫu


print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
model = RWKV_RNN(args)

init_state = None
for token in tokenizer(context)["input_ids"]:
    init_out, init_state = model.forward(token, init_state)

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out.clone(), init_state.clone()
    for i in range(LENGTH_PER_TRIAL):
        token = sample_logits(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        tmp = tokenizer.decode(all_tokens[out_last:])
        print(tmp, end="", flush=True)
        out_last = i + 1
        out, state = model.forward(token, state)       
print('\n')
