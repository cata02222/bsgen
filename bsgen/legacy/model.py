import torch
from torch import nn
from torch.nn import functional as F

from .data import MAX_TOKENS, N_FRAMES, OVERVIEW_BARS
from .tokens import COND_FIELDS


class ConvBlock(nn.Module):

    def __init__(self, d, kernel=9, dilation=1, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, kernel, padding=dilation * (kernel // 2), dilation=dilation, groups=d)
        self.pw = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.pw(self.dw(self.norm(x).transpose(1, 2)).transpose(1, 2))


class Generator(nn.Module):
    def __init__(self, vocab: int, d=384, enc_layers=6, dec_layers=6, heads=6, n_mels=80, dropout=0.1,
                 max_tokens=MAX_TOKENS, conv_blocks=0, global_cond=False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_mels, d, 3, padding=1), nn.GELU(), nn.Conv1d(d, d, 4, stride=2, padding=1))
        self.convs = nn.Sequential(*(ConvBlock(d, dilation=2 ** (i % 3), dropout=dropout) for i in range(conv_blocks)))
        self.enc_pos = nn.Parameter(torch.randn(N_FRAMES // 2, d) * 0.02)
        self.ov_in = nn.Linear(2 * n_mels, d)
        self.ov_rel = nn.Embedding(2 * OVERVIEW_BARS - 1, d)
        nn.init.normal_(self.ov_rel.weight, std=0.02)
        layer = lambda: dict(d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=dropout,
                             activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(**layer()), enc_layers, enable_nested_tensor=False)
        self.tok = nn.Embedding(vocab, d)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.dec_pos = nn.Parameter(torch.randn(max_tokens, d) * 0.02)
        self.cond = None
        if global_cond:
            n = len(COND_FIELDS)
            self.cond = nn.Sequential(nn.Linear(n * d, d), nn.GELU(), nn.Linear(d, d))
            nn.init.zeros_(self.cond[2].weight)
            nn.init.zeros_(self.cond[2].bias)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(**layer()), dec_layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)
        self.out.weight = self.tok.weight

    def encode(self, mel, overview, rel):
        x = self.convs(self.stem(mel.transpose(1, 2)).transpose(1, 2))
        x = x + self.enc_pos[: x.shape[1]]
        o = self.ov_in(overview) + self.ov_rel(rel)
        return self.enc(torch.cat([x, o], 1))

    def global_cond(self, tk):
        if self.cond is None:
            return 0
        c = self.tok(tk[:, 1:1 + len(COND_FIELDS)])
        return self.cond(c.flatten(1))[:, None]

    def decode(self, memory, tk):
        L = tk.shape[1]
        x = self.tok(tk) + self.dec_pos[:L] + self.global_cond(tk)
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=tk.device)
        return self.out(self.norm(self.dec(x, memory, tgt_mask=causal, tgt_is_causal=True)))

    def decode_step(self, memory, tk, cache):
        past = cache.get("len", 0)
        if past == 0:
            cache["g"] = self.global_cond(tk)
        x = self.tok(tk) + self.dec_pos[past:past + tk.shape[1]] + cache["g"]
        for i, layer in enumerate(self.dec.layers):
            c = cache.setdefault(i, {})
            sa, ca = layer.self_attn, layer.multihead_attn
            q, k, v = self._heads(F.linear(layer.norm1(x), sa.in_proj_weight, sa.in_proj_bias), sa.num_heads, 3)
            if "k" in c:
                k, v = torch.cat([c["k"], k], 2), torch.cat([c["v"], v], 2)
            c["k"], c["v"] = k, v
            if tk.shape[1] == 1:
                a = self._attend(q, k, v)
            else:
                a = F.scaled_dot_product_attention(q, k, v, is_causal=past == 0)
            x = x + sa.out_proj(self._merge(a))
            d = ca.embed_dim
            if "mk" not in c:
                c["mk"], c["mv"] = self._heads(F.linear(memory, ca.in_proj_weight[d:], ca.in_proj_bias[d:]), ca.num_heads, 2)
            q, = self._heads(F.linear(layer.norm2(x), ca.in_proj_weight[:d], ca.in_proj_bias[:d]), ca.num_heads, 1)
            x = x + ca.out_proj(self._merge(self._cross(q, c["mk"], c["mv"])))
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))
        cache["len"] = past + tk.shape[1]
        return self.out(self.norm(x[:, -1]))

    @staticmethod
    def _attend(q, k, v, mask=None):
        w = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
        if mask is not None:
            w = w.masked_fill(~mask, float("-inf"))
        return w.softmax(-1) @ v

    @staticmethod
    def _cross(q, mk, mv):
        if mk.shape[0] == q.shape[0]:
            return F.scaled_dot_product_attention(q, mk, mv)
        B, h, T, dh = q.shape
        q = q.transpose(0, 1).reshape(1, h, B * T, dh)
        o = Generator._attend(q, mk, mv) if T == 1 else F.scaled_dot_product_attention(q, mk, mv)
        return o.view(h, B, T, dh).transpose(0, 1)

    @staticmethod
    def _heads(y, h, n):
        B, T, _ = y.shape
        return y.view(B, T, n, h, -1).permute(2, 0, 3, 1, 4).unbind(0)

    @staticmethod
    def _merge(y):
        B, h, T, dh = y.shape
        return y.transpose(1, 2).reshape(B, T, h * dh)

    def forward(self, audio, tk):
        return self.decode(self.encode(*audio), tk)

    def loss(self, audio, tk, mask, per_window=False):
        logits = self(audio, tk[:, :-1])
        m = mask[:, 1:]
        ce = F.cross_entropy(logits.transpose(1, 2), tk[:, 1:], reduction="none")
        if per_window:
            n = m.sum(1)
            return ((ce * m).sum(1) / n.clamp(min=1))[n > 0].mean()
        return (ce * m).sum() / m.sum().clamp(min=1)


class GraphStep:

    BUCKET = 128

    def __init__(self, model, batch, mem_len):
        self.m = model
        L, d = model.dec_pos.shape
        h = model.dec.layers[0].self_attn.num_heads
        dev = model.dec_pos.device
        buf = lambda n: torch.zeros(batch, h, n, d // h, device=dev)
        self.k = [buf(L) for _ in model.dec.layers]
        self.v = [buf(L) for _ in model.dec.layers]
        mem = lambda: torch.zeros(1, h, mem_len, d // h, device=dev)
        self.mk = [mem() for _ in model.dec.layers]
        self.mv = [mem() for _ in model.dec.layers]
        self.g = torch.zeros(batch, 1, d, device=dev)
        self.tok = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.ar = torch.arange(L, device=dev)
        self.len = 0
        self.graphs, self.outs = {}, {}

    def prefill(self, memory, tk):
        cache = {}
        lg = self.m.decode_step(memory, tk, cache)
        T = tk.shape[1]
        for i in range(len(self.k)):
            c = cache[i]
            self.k[i][:, :, :T].copy_(c["k"])
            self.v[i][:, :, :T].copy_(c["v"])
            self.mk[i].copy_(c["mk"])
            self.mv[i].copy_(c["mv"])
        self.g.copy_(cache["g"] if torch.is_tensor(cache["g"]) else torch.zeros_like(self.g))
        self.len = T
        return lg

    def _body(self, n):
        m = self.m
        x = m.tok(self.tok) + m.dec_pos.index_select(0, self.pos)[None] + self.g
        visible = self.ar[:n] <= self.pos
        for i, layer in enumerate(m.dec.layers):
            sa, ca = layer.self_attn, layer.multihead_attn
            q, k, v = m._heads(F.linear(layer.norm1(x), sa.in_proj_weight, sa.in_proj_bias), sa.num_heads, 3)
            self.k[i].index_copy_(2, self.pos, k)
            self.v[i].index_copy_(2, self.pos, v)
            x = x + sa.out_proj(m._merge(m._attend(q, self.k[i][:, :, :n], self.v[i][:, :, :n], visible)))
            d = ca.embed_dim
            q, = m._heads(F.linear(layer.norm2(x), ca.in_proj_weight[:d], ca.in_proj_bias[:d]), ca.num_heads, 1)
            x = x + ca.out_proj(m._merge(m._cross(q, self.mk[i], self.mv[i])))
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))
        return m.out(m.norm(x[:, -1]))

    def capture(self):
        pool = None
        for n in range(self.BUCKET, len(self.ar) + self.BUCKET, self.BUCKET):
            n = min(n, len(self.ar))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._body(n)
            torch.cuda.current_stream().wait_stream(s)
            self.graphs[n] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graphs[n], pool=pool):
                self.outs[n] = self._body(n)
            pool = self.graphs[n].pool()
        torch.cuda.synchronize()

    def step(self, tk):
        self.tok.copy_(tk)
        self.pos.fill_(self.len)
        self.len += 1
        n = min(-(-self.len // self.BUCKET) * self.BUCKET, len(self.ar))
        self.graphs[n].replay()
        return self.outs[n]
