import threading

import torch
from torch import nn
from torch.nn import functional as F

from .data import FRAMES_PER_BEAT, HIT_DIM, MAX_TOKENS, MEM_SLOTS, N_FRAMES, OVERVIEW_BARS, ROLL_DIM
from .tokens import COND_FIELDS, CONTEXT

RECORDING = threading.Lock()


def recording_stream():
    return torch.cuda.Stream(priority=-1)


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
                 max_tokens=MAX_TOKENS, conv_blocks=0, global_cond=False, memory=False, mem_align=False,
                 per_beat=FRAMES_PER_BEAT, hits=False):
        super().__init__()
        self.per_beat = per_beat
        s = per_beat // 6
        self.stem = nn.Sequential(nn.Conv1d(n_mels, d, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(d, d, 2 * s, stride=s, padding=s // 2))
        self.convs = nn.Sequential(*(ConvBlock(d, dilation=2 ** (i % 3), dropout=dropout) for i in range(conv_blocks)))
        self.enc_pos = nn.Parameter(torch.randn(N_FRAMES // 2, d) * 0.02)
        self.ov_in = nn.Linear(2 * n_mels, d)
        self.ov_rel = nn.Embedding(2 * OVERVIEW_BARS - 1, d)
        nn.init.normal_(self.ov_rel.weight, std=0.02)
        self.memory, self.mem_align = memory, mem_align
        if memory:
            if not mem_align:
                self.mem_pos = nn.Parameter(torch.randn(MEM_SLOTS, d) * 0.02)
            self.mem_roll = nn.Linear(ROLL_DIM, d)
            self.mem_sim = nn.Linear(2, d)
            for lin in (self.mem_roll, self.mem_sim):
                nn.init.normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)
        self.hits = hits
        if hits:
            self.hit_in = nn.Linear(HIT_DIM, d)
            nn.init.zeros_(self.hit_in.weight)
            nn.init.zeros_(self.hit_in.bias)
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
        self.step_dec = None

    def half_steps(self):
        import copy
        self.step_dec = copy.deepcopy(self.dec).to(torch.bfloat16)

    def _step_layers(self):
        dec = self.step_dec if self.step_dec is not None else self.dec
        return dec.layers, dec.layers[0].linear1.weight.dtype

    def frames(self, mel):
        return self.convs(self.stem(mel.transpose(1, 2)).transpose(1, 2))

    def encode(self, mel, overview, rel, *rest):
        mem_mel, roll, sim = rest[:3] if self.memory else (None, None, None)
        x = self.frames(mel)
        if self.hits:
            x = x + self.hit_in(rest[-1])
        if self.memory and self.mem_align:
            a = CONTEXT * FRAMES_PER_BEAT // 2
            mem = self.mem_roll(roll) + self.mem_sim(sim)[:, None]
            x = torch.cat([x[:, :a], x[:, a:a + MEM_SLOTS] + mem, x[:, a + MEM_SLOTS:]], 1)
        parts = [x + self.enc_pos[:x.shape[1]], self.ov_in(overview) + self.ov_rel(rel)]
        if self.memory and not self.mem_align:
            parts.append(self.frames(mem_mel) + self.mem_pos + self.mem_roll(roll) + self.mem_sim(sim)[:, None])
        return self.enc(torch.cat(parts, 1))

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
        layers, dt = self._step_layers()
        x = (self.tok(tk) + self.dec_pos[past:past + tk.shape[1]] + cache["g"]).to(dt)
        memory = memory.to(dt)
        for i, layer in enumerate(layers):
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
        return self.out(self.norm(x[:, -1].float()))

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
        self.layers, self.dt = model._step_layers()
        h = self.layers[0].self_attn.num_heads
        dev = model.dec_pos.device
        buf = lambda n: torch.zeros(batch, h, n, d // h, device=dev, dtype=self.dt)
        self.k = [buf(L) for _ in self.layers]
        self.v = [buf(L) for _ in self.layers]
        mem = lambda: torch.zeros(1, h, mem_len, d // h, device=dev, dtype=self.dt)
        self.mk = [mem() for _ in self.layers]
        self.mv = [mem() for _ in self.layers]
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
        x = (m.tok(self.tok) + m.dec_pos.index_select(0, self.pos)[None] + self.g).to(self.dt)
        visible = self.ar[:n] <= self.pos
        for i, layer in enumerate(self.layers):
            sa, ca = layer.self_attn, layer.multihead_attn
            q, k, v = m._heads(F.linear(layer.norm1(x), sa.in_proj_weight, sa.in_proj_bias), sa.num_heads, 3)
            self.k[i].index_copy_(2, self.pos, k)
            self.v[i].index_copy_(2, self.pos, v)
            x = x + sa.out_proj(m._merge(m._attend(q, self.k[i][:, :, :n], self.v[i][:, :, :n], visible)))
            d = ca.embed_dim
            q, = m._heads(F.linear(layer.norm2(x), ca.in_proj_weight[:d], ca.in_proj_bias[:d]), ca.num_heads, 1)
            x = x + ca.out_proj(m._merge(m._cross(q, self.mk[i], self.mv[i])))
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))
        return m.out(m.norm(x[:, -1].float()))

    def capture(self):
        with RECORDING:
            self._capture()

    def _capture(self):
        pool, rec = None, recording_stream()
        for n in range(self.BUCKET, len(self.ar) + self.BUCKET, self.BUCKET):
            n = min(n, len(self.ar))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._body(n)
            torch.cuda.current_stream().wait_stream(s)
            self.graphs[n] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graphs[n], pool=pool, stream=rec, capture_error_mode="thread_local"):
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


class SharedStep(GraphStep):

    def __init__(self, model, groups, per, mem_len):
        rows = groups * per
        super().__init__(model, rows, mem_len)
        L, d = model.dec_pos.shape
        h = self.layers[0].self_attn.num_heads
        dev = model.dec_pos.device
        self.groups, self.per = groups, per
        self.k = [torch.zeros(rows, h, L + 1, d // h, device=dev, dtype=self.dt) for _ in self.layers]
        self.v = [torch.zeros(rows, h, L + 1, d // h, device=dev, dtype=self.dt) for _ in self.layers]
        self.mk = [torch.zeros(groups, h, mem_len, d // h, device=dev, dtype=self.dt) for _ in self.layers]
        self.mv = [torch.zeros(groups, h, mem_len, d // h, device=dev, dtype=self.dt) for _ in self.layers]
        self.pos = torch.full((rows,), L, dtype=torch.long, device=dev)
        vocab = model.out.weight.shape[0]
        self.tok_host = torch.zeros(rows, 1, dtype=torch.long).pin_memory()
        self.pos_host = torch.full((rows,), L, dtype=torch.long).pin_memory()
        self.out_host = torch.zeros(rows, vocab).pin_memory()
        self.tok_np, self.pos_np = self.tok_host.numpy(), self.pos_host.numpy()
        self.cv = threading.Condition()
        self.free, self.holders, self.asked = list(range(groups)), set(), set()
        self.at = {}
        self.round, self.done = 0, None
        self.stream = torch.cuda.Stream(dev)

    def _body(self, n):
        m, L = self.m, self.m.dec_pos.shape[0]
        R = self.tok.shape[0]
        x = (m.tok(self.tok) + m.dec_pos.index_select(0, self.pos.clamp(max=L - 1))[:, None] + self.g).to(self.dt)
        visible = (self.ar[:n][None] <= self.pos[:, None])[:, None, None]
        for i, layer in enumerate(self.layers):
            sa, ca = layer.self_attn, layer.multihead_attn
            q, k, v = m._heads(F.linear(layer.norm1(x), sa.in_proj_weight, sa.in_proj_bias), sa.num_heads, 3)
            at = self.pos.view(R, 1, 1, 1).expand(k.shape)
            self.k[i].scatter_(2, at, k)
            self.v[i].scatter_(2, at, v)
            a = F.scaled_dot_product_attention(q, self.k[i][:, :, :n], self.v[i][:, :, :n], attn_mask=visible)
            x = x + sa.out_proj(m._merge(a))
            d = ca.embed_dim
            q, = m._heads(F.linear(layer.norm2(x), ca.in_proj_weight[:d], ca.in_proj_bias[:d]), ca.num_heads, 1)
            dh = q.shape[-1]
            q = q.view(self.groups, self.per, ca.num_heads, dh).transpose(1, 2)
            o = m._attend(q, self.mk[i], self.mv[i]).transpose(1, 2).reshape(R, ca.num_heads, 1, dh)
            x = x + ca.out_proj(m._merge(o))
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))
        return m.out(m.norm(x[:, -1].float()))

    def take(self, k):
        with self.cv:
            if k > self.per or not self.free:
                return None
            g = self.free.pop(0)
            return range(g * self.per, g * self.per + k)

    def join(self, rows):
        with self.cv:
            self.holders.add(rows)

    def give(self, rows):
        with self.cv:
            self.holders.discard(rows)
            self.asked.discard(rows)
            self.pos_np[rows.start:rows.stop] = self.m.dec_pos.shape[0]
            self.free.append(rows.start // self.per)
            self._maybe_run()

    def prefill(self, rows, memory, tk):
        cache, mine = {}, torch.cuda.current_stream()
        self.stream.wait_stream(mine)
        with torch.cuda.stream(self.stream):
            lg = self.m.decode_step(memory, tk, cache)
            T, at, g = tk.shape[1], slice(rows.start, rows.stop), rows.start // self.per
            for i in range(len(self.k)):
                c = cache[i]
                self.k[i][at, :, :T] = c["k"]
                self.v[i][at, :, :T] = c["v"]
                self.mk[i][g] = c["mk"][0]
                self.mv[i][g] = c["mv"][0]
            self.g[at] = cache["g"] if torch.is_tensor(cache["g"]) else 0
        memory.record_stream(self.stream)
        tk.record_stream(self.stream)
        lg.record_stream(mine)
        mine.wait_stream(self.stream)
        return lg

    def ask(self, rows, tokens, pos):
        with self.cv:
            self.tok_np[rows.start:rows.stop, 0] = tokens
            self.pos_np[rows.start:rows.stop] = pos
            self.at[rows] = pos
            self.asked.add(rows)
            mine = self.round
            self._maybe_run()
            return mine

    def _maybe_run(self):
        if not self.holders or not self.holders <= self.asked:
            return
        top = max(self.at[r] for r in self.holders) + 1
        n = min(-(-top // self.BUCKET) * self.BUCKET, len(self.ar))
        with torch.cuda.stream(self.stream):
            self.tok.copy_(self.tok_host, non_blocking=True)
            self.pos.copy_(self.pos_host, non_blocking=True)
            self.graphs[n].replay()
            self.out_host.copy_(self.outs[n], non_blocking=True)
            self.done = torch.cuda.Event()
            self.done.record()
        self.round += 1
        self.asked.clear()
        self.cv.notify_all()

    def result(self, rows, mine):
        with self.cv:
            self.cv.wait_for(lambda: self.round > mine)
            done = self.done
        done.synchronize()
        return self.out_host[rows.start:rows.stop].clone()
