import torch
import torch.nn as nn


class ExogenousCorrelationGate(nn.Module):
    def __init__(self, num_aux_features, hidden_dim, gate_scale, eps=1e-6):
        super().__init__()
        self.num_aux_features = int(num_aux_features)
        self.gate_scale = float(gate_scale)
        self.eps = float(eps)
        hidden = max(8, int(hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(self.num_aux_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_aux_features),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x_main, x_aux):
        if self.num_aux_features <= 0:
            return x_aux
        power = x_main - x_main.mean(dim=1, keepdim=True)
        aux = x_aux - x_aux.mean(dim=1, keepdim=True)
        numerator = (power * aux).sum(dim=1)
        power_norm = torch.sqrt(power.pow(2).sum(dim=1) + self.eps)
        aux_norm = torch.sqrt(aux.pow(2).sum(dim=1) + self.eps)
        corr = numerator / (power_norm * aux_norm + self.eps)
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        delta = torch.tanh(self.mlp(corr)) * self.gate_scale
        return x_aux * (1.0 + delta.unsqueeze(1))


class SelectiveSegmentGate(nn.Module):
    def __init__(self, d_model, init_alpha):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 2 * d_model),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, z):
        input_raw, forget_raw = self.gate(self.norm(z)).chunk(2, dim=-1)
        input_gate = torch.sigmoid(input_raw)
        forget_gate = torch.sigmoid(forget_raw)
        mod = input_gate * torch.tanh(z) + forget_gate * z
        return z + self.alpha * (mod - z)


class ChannelAttention(nn.Module):
    def __init__(self, d_model, dropout, num_heads, init_alpha):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, h):
        S, C, m, d = h.shape
        if C <= 1:
            return h
        z = h.permute(0, 2, 1, 3).reshape(S * m, C, d)
        zn = self.norm(z)
        a, _ = self.attn(zn, zn, zn)
        z = z + self.alpha * a
        return z.reshape(S, m, C, d).permute(0, 2, 1, 3).contiguous()


class RegimePrototypeMemory(nn.Module):
    def __init__(self, d_model, num_prototypes, init_alpha):
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, d_model) * 0.02)
        self.value = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        self.norm = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, h):
        z = self.norm(h)
        logits = torch.matmul(z, self.prototypes.t()) / (z.shape[-1] ** 0.5)
        weight = torch.softmax(logits, dim=-1)
        memory = torch.matmul(weight, self.prototypes)
        return h + self.alpha * self.value(memory)


class UnitConditioning(nn.Module):
    def __init__(self, num_units, d_model, prompt_alpha, film_alpha):
        super().__init__()
        self.d_model = int(d_model)
        self.emb = nn.Embedding(int(num_units), 3 * self.d_model)
        nn.init.zeros_(self.emb.weight)
        self.prompt_alpha = nn.Parameter(torch.tensor(float(prompt_alpha)))
        self.film_alpha = nn.Parameter(torch.tensor(float(film_alpha)))

    def forward(self, h, unit_idx):
        e = self.emb(unit_idx.long())
        prompt, scale, shift = e.chunk(3, dim=-1)
        prompt = prompt.unsqueeze(1)
        scale = torch.tanh(scale).unsqueeze(1)
        shift = shift.unsqueeze(1)
        h = h + self.prompt_alpha * prompt
        h = h + self.film_alpha * (h * scale + shift)
        return h


class UnitAttention(nn.Module):
    def __init__(self, d_model, dropout, num_heads, init_alpha):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, h):
        B, K, m, d = h.shape
        if K <= 1:
            return h
        z = h.permute(0, 2, 1, 3).reshape(B * m, K, d)
        zn = self.norm(z)
        a, _ = self.attn(zn, zn, zn)
        z = z + self.alpha * a
        return z.reshape(B, m, K, d).permute(0, 2, 1, 3).contiguous()


class Model(nn.Module):
    supports_dynamic_units = True

    def __init__(self, config, channel, num_units):
        super().__init__()
        self.p = int(config.hist_len)
        self.q = int(config.pred_len)
        self.num_features = int(channel)
        self.num_units = int(num_units)
        self.output_dim = int(config.output_dim)
        self.d_model = int(config.d_model)
        self.dropout = float(config.dropout)
        self.w = int(config.seg_len)
        if self.output_dim != 1:
            raise ValueError('DC-SDPNet requires output_dim=1.')
        if self.p % self.w != 0 or self.q % self.w != 0:
            raise ValueError(f'seg_len={self.w} must divide hist_len={self.p} and pred_len={self.q}.')
        if self.d_model % 2 != 0:
            raise ValueError('d_model must be even.')
        if self.d_model % int(config.num_heads) != 0:
            raise ValueError('d_model must be divisible by num_heads.')
        self.n = self.p // self.w
        self.m = self.q // self.w
        self.seg_len = self.w
        self.seg_num_x = self.n
        self.seg_num_y = self.m
        self.num_aux_features = max(0, self.num_features - 1)
        self.anchor_value = nn.Sequential(nn.Linear(self.w, self.d_model), nn.ReLU())
        self.anchor_rnn = nn.GRU(self.d_model, self.d_model, num_layers=1, bias=True, batch_first=True)
        self.anchor_pos_emb = nn.Parameter(torch.randn(self.m, self.d_model // 2))
        self.anchor_channel_emb = nn.Parameter(torch.randn(1, self.d_model // 2))
        self.anchor_predict = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.d_model, self.w))
        self.corr_gate = (
            ExogenousCorrelationGate(
                self.num_aux_features,
                hidden_dim=max(16, self.d_model // 2),
                gate_scale=config.corr_gate_scale,
            )
            if self.num_aux_features > 0
            else None
        )
        self.res_value = nn.Sequential(nn.Linear(self.w, self.d_model), nn.ReLU())
        self.selective_gate = SelectiveSegmentGate(self.d_model, config.selective_gate_alpha)
        self.res_rnn = nn.GRU(self.d_model, self.d_model, num_layers=1, bias=True, batch_first=True)
        self.res_pos_emb = nn.Parameter(torch.randn(self.m, self.d_model // 2))
        self.res_channel_emb = nn.Parameter(torch.randn(self.num_features, self.d_model // 2))
        self.channel_attn = ChannelAttention(
            self.d_model,
            dropout=self.dropout,
            num_heads=config.num_heads,
            init_alpha=config.channel_attn_alpha,
        )
        self.regime_memory = RegimePrototypeMemory(
            self.d_model,
            num_prototypes=config.num_prototypes,
            init_alpha=config.regime_alpha,
        )
        self.unit_conditioning = UnitConditioning(
            self.num_units,
            self.d_model,
            prompt_alpha=config.prompt_alpha,
            film_alpha=config.film_alpha,
        )
        self.unit_attn = UnitAttention(
            self.d_model,
            dropout=self.dropout,
            num_heads=config.num_heads,
            init_alpha=config.unit_attn_alpha,
        )
        self.context_gate = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 1))
        nn.init.zeros_(self.context_gate[-1].weight)
        nn.init.constant_(self.context_gate[-1].bias, float(config.context_gate_bias))
        self.res_predict = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.d_model, self.w))
        self.residual_alpha = nn.Parameter(torch.tensor(float(config.residual_alpha_init)))

    def _anchor_forward(self, x_main):
        S = x_main.shape[0]
        last = x_main[:, -1:, :].detach()
        x = (x_main - last).transpose(1, 2).reshape(S, self.n, self.w)
        z = self.anchor_value(x)
        _, hn = self.anchor_rnn(z)
        pmf_query = torch.cat(
            [self.anchor_pos_emb, self.anchor_channel_emb.repeat(self.m, 1)],
            dim=-1,
        ).view(1, self.m, self.d_model).repeat(S, 1, 1)
        hn_rep = hn.repeat(1, 1, self.m).view(1, -1, self.d_model)
        _, hy = self.anchor_rnn(pmf_query.reshape(S * self.m, 1, self.d_model), hn_rep)
        h = hy.squeeze(0).view(S, self.m, self.d_model)
        y_delta = self.anchor_predict(h).reshape(S, self.q, 1)
        return y_delta + last, h

    def _build_residual_input(self, x_flat):
        x_main = x_flat[..., :1]
        if self.num_features <= 1:
            return x_main
        x_aux = x_flat[..., 1:]
        if self.corr_gate is not None:
            x_aux = self.corr_gate(x_main, x_aux)
        return torch.cat([x_main, x_aux], dim=-1)

    def _residual_forward(self, x_flat, unit_flat, B, K):
        S, _, _ = x_flat.shape
        x_core = self._build_residual_input(x_flat)
        C = x_core.shape[-1]
        last = x_core[:, -1:, :].detach()
        x = (x_core - last).permute(0, 2, 1).reshape(S * C, self.n, self.w)
        z = self.res_value(x)
        z = self.selective_gate(z)
        _, hn = self.res_rnn(z)
        pmf_query = torch.cat(
            [
                self.res_pos_emb.unsqueeze(0).repeat(C, 1, 1),
                self.res_channel_emb[:C].unsqueeze(1).repeat(1, self.m, 1),
            ],
            dim=-1,
        ).view(-1, 1, self.d_model).repeat(S, 1, 1)
        hn_rep = hn.repeat(1, 1, self.m).view(1, -1, self.d_model)
        _, hy = self.res_rnn(pmf_query, hn_rep)
        h_all = hy.squeeze(0).view(S, C, self.m, self.d_model)
        h_all = self.channel_attn(h_all)
        h_target = h_all[:, 0, :, :]
        h_target = self.regime_memory(h_target)
        h_target = self.unit_conditioning(h_target, unit_flat)
        h4 = h_target.reshape(B, K, self.m, self.d_model)
        h_target = self.unit_attn(h4).reshape(S, self.m, self.d_model)
        gate = torch.sigmoid(self.context_gate(h_target))
        y_seg = self.res_predict(h_target)
        y_res = (gate * y_seg).reshape(S, self.q, 1)
        return self.residual_alpha * y_res

    def forward(self, x_data, unit_idx):
        B, K, L, N_in = x_data.shape
        if L != self.p:
            raise ValueError(f'Expected history length p={self.p}, got L={L}.')
        if N_in != self.num_features:
            raise ValueError(f'Expected feature dimension N={self.num_features}, got N_in={N_in}.')
        x_flat = x_data.reshape(B * K, L, N_in)
        unit_flat = unit_idx.reshape(B * K).long().clamp(min=0, max=self.num_units - 1)
        x_main = x_flat[..., :1]
        y_anchor, _ = self._anchor_forward(x_main)
        y_res = self._residual_forward(x_flat, unit_flat, B, K)
        y = y_anchor + y_res
        return y.reshape(B, K, self.q, 1)
