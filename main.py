import sympy as sp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random, re, json, math, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter, defaultdict

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

"""
We frame the problem as machine translation:
    source = mathematical function        e.g.  sin(x) + x**2
    target = 4th-order Taylor series      e.g.  x**2 + x - x**3/6

SymPy computes Taylor series analytically:
    sp.series(f, x, 0, 5).removeO()
    → exact polynomial up to x^4, no floating-point error.

Function type labels use 'X' (capital X) as the scalar suffix separator
to avoid clashing with the '+' separator used in compound names, and to
stay fully ASCII-safe. The base_type() helper correctly strips all suffixes.
"""

def generate_dataset(n_samples=3000, seed=42):
    """
    Returns
    -------
    pairs      : list of (input_str, target_str)
    func_types : list of type labels, one per pair
    """
    random.seed(seed); np.random.seed(seed)
    x = sp.Symbol('x')

    base_funcs = {
        'sin':  lambda: sp.sin(x),
        'cos':  lambda: sp.cos(x),
        'exp':  lambda: sp.exp(x),
        'log':  lambda: sp.log(1 + x),
        'tan':  lambda: sp.tan(x),
        'sinh': lambda: sp.sinh(x),
        'cosh': lambda: sp.cosh(x),
        'atan': lambda: sp.atan(x),
        'inv':  lambda: 1 / (1 - x),
        'sqrt': lambda: (1 + x) ** sp.Rational(1, 2),
    }

    def rand_poly():
        d = random.randint(1, 3)
        return sum(random.randint(-3, 3) * x**i for i in range(d + 1))

    pairs, func_types = [], []
    attempts = 0

    while len(pairs) < n_samples and attempts < n_samples * 15:
        attempts += 1
        try:
            r     = random.random()
            fname = random.choice(list(base_funcs.keys()))
            f     = base_funcs[fname]()

            if   r < 0.28:
                ftype = fname
            elif r < 0.50:
                f = f + rand_poly()
                ftype = fname + '+poly'
            elif r < 0.65:
                f = random.randint(1, 3) * f
                ftype = fname + 'Xscalar'   # capital X — no clash with base names
            elif r < 0.80:
                f2    = random.choice(list(base_funcs.keys()))
                f     = f + base_funcs[f2]()
                ftype = fname + '+' + f2
            else:
                f     = rand_poly()
                ftype = 'poly'

            taylor = sp.series(f, x, 0, 5).removeO()
            taylor = sp.simplify(taylor)
            inp, tgt = str(f), str(taylor)

            if inp and tgt and len(inp) < 80 and len(tgt) < 80:
                pairs.append((inp, tgt))
                func_types.append(ftype)
        except Exception:
            continue

    print(f"  Generated {len(pairs)} (function, Taylor expansion) pairs")
    print(f"  Examples:")
    for i, t in pairs[:4]:
        print(f"    f(x) = {i:<38}  T[f] = {t}")
    return pairs, func_types


def base_type(ftype):
    """
    Extract base function name from a compound type label.
    'sin+poly' → 'sin', 'sinXscalar' → 'sin', 'sin+cos' → 'sin', 'poly' → 'poly'
    Split on '+' first, then on 'X' (our scalar marker).
    """
    return ftype.split('+')[0].split('X')[0]


"""
Mathematical expressions are split into atomic tokens:
    "sin(x) + x**2/2" → ['sin', '(', 'x', ')', '+', 'x', '**', '2', '/', '2']

Special tokens
    <PAD> (0)  padding to equal length
    <SOS> (1)  start-of-sequence
    <EOS> (2)  end-of-sequence
    <UNK> (3)  unknown token (safety fallback)
"""

def tokenize(expr):
    return re.findall(
        r'(\d+/\d+|\d+\.\d+|\d+|[a-zA-Z_]\w*|\*\*|[+\-*/(),])', expr)


def normalize(expr):
    """
    Canonical string form: tokenize then rejoin with no whitespace.
    'x**3/6 + x'  →  'x**3/6+x'
    This ensures that SymPy's spacing style does not cause false
    mismatches when comparing predicted vs reference expressions.
    Used in per_functype_exact() and greedy_str() comparisons.
    """
    return ''.join(tokenize(expr))


class Vocabulary:
    PAD, SOS, EOS, UNK = '<PAD>', '<SOS>', '<EOS>', '<UNK>'

    def __init__(self):
        self.token2idx = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        self.idx2token = {v: k for k, v in self.token2idx.items()}

    def build(self, token_lists):
        for tokens in token_lists:
            for t in tokens:
                if t not in self.token2idx:
                    i = len(self.token2idx)
                    self.token2idx[t] = i
                    self.idx2token[i] = t

    def encode(self, tokens, max_len=50):
        ids = [self.token2idx[self.SOS]]
        for t in tokens[:max_len - 2]:
            ids.append(self.token2idx.get(t, self.token2idx[self.UNK]))
        ids.append(self.token2idx[self.EOS])
        ids += [self.token2idx[self.PAD]] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            t = self.idx2token.get(i, self.UNK)
            if t == self.EOS: break
            if t not in (self.PAD, self.SOS): out.append(t)
        return ''.join(out)

    def __len__(self): return len(self.token2idx)


class TaylorDataset(Dataset):
    def __init__(self, pairs, sv, tv, max_len=50):
        self.data = [
            (torch.tensor(sv.encode(tokenize(i), max_len), dtype=torch.long),
             torch.tensor(tv.encode(tokenize(t), max_len), dtype=torch.long))
            for i, t in pairs]

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

"""
Classic encoder-decoder (Sutskever et al., 2014).
Encoder compresses the full input into one fixed context vector (h, c).
Decoder generates tokens conditioned on that single vector.

Known limitation: everything must fit in one vector → information bottleneck
→ val loss diverges from train loss after a few epochs (overfitting signal).
We fix this in Model B with Bahdanau attention.
"""

class VanillaEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        emb = self.dropout(self.embedding(src))
        _, (h, c) = self.lstm(emb)
        return h, c


class VanillaDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, h, c):
        emb = self.dropout(self.embedding(token.unsqueeze(1)))
        out, (h, c) = self.lstm(emb, (h, c))
        return self.fc_out(out.squeeze(1)), h, c


class VanillaSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder; self.decoder = decoder; self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        B, T = tgt.shape
        vs   = self.decoder.fc_out.out_features
        outs = torch.zeros(B, T, vs).to(self.device)
        h, c = self.encoder(src)
        dec  = tgt[:, 0]
        for t in range(1, T):
            pred, h, c = self.decoder(dec, h, c)
            outs[:, t] = pred
            dec = tgt[:, t] if random.random() < teacher_forcing_ratio \
                  else pred.argmax(1)
        return outs

"""
Bahdanau (additive) attention (Bahdanau et al., 2015).
Replaces the single fixed context vector with a dynamic weighted sum
over ALL encoder hidden states at each decoding step:

    score(s_t, h_s) = v^T tanh(W1 s_t + W2 h_s)
    alpha_t         = softmax(scores)           attention weights
    context_t       = sum_s alpha_{t,s} h_s    dynamic context

The context is concatenated with the token embedding as LSTM input,
and again with the LSTM output before the final projection.
This directly eliminates the information bottleneck.
"""

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v  = nn.Linear(hidden_dim, 1,          bias=False)

    def forward(self, dec_h, enc_out):
        # dec_h: (B,H)  enc_out: (B,S,H)
        scores  = self.v(torch.tanh(self.W1(dec_h.unsqueeze(1)) + self.W2(enc_out)))
        weights = torch.softmax(scores, dim=1)     # (B,S,1)
        context = (weights * enc_out).sum(1)       # (B,H)
        return context, weights.squeeze(-1)        # (B,S)


class AttentionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        emb = self.dropout(self.embedding(src))
        outputs, (h, c) = self.lstm(emb)
        return outputs, h, c   # return all encoder outputs for attention


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, h, c, enc_out):
        emb             = self.dropout(self.embedding(token.unsqueeze(1)))
        context, attn_w = self.attention(h[-1], enc_out)
        lstm_in         = torch.cat([emb, context.unsqueeze(1)], dim=-1)
        out, (h, c)     = self.lstm(lstm_in, (h, c))
        pred = self.fc_out(torch.cat([out.squeeze(1), context], dim=-1))
        return pred, h, c, attn_w


class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder; self.decoder = decoder; self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        B, T = tgt.shape
        vs   = self.decoder.fc_out.out_features
        outs = torch.zeros(B, T, vs).to(self.device)
        enc_out, h, c = self.encoder(src)
        dec = tgt[:, 0]
        for t in range(1, T):
            pred, h, c, _ = self.decoder(dec, h, c, enc_out)
            outs[:, t] = pred
            dec = tgt[:, t] if random.random() < teacher_forcing_ratio \
                  else pred.argmax(1)
        return outs

"""
Transformer (Vaswani et al., 2017). Multi-head self-attention processes
all tokens in parallel — no sequential bottleneck, better long-range
dependency modelling, faster convergence.

Adam betas=(0.9, 0.98), eps=1e-9 follow the original paper.
ReduceLROnPlateau halves LR when val loss stops improving.
"""

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                             (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vs, tgt_vs, embed_dim=256, n_heads=8,
                 ff_dim=512, n_layers=3, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.src_embed = nn.Embedding(src_vs, embed_dim, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vs, embed_dim, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(embed_dim, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=n_heads,
            num_encoder_layers=n_layers, num_decoder_layers=n_layers,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.fc_out  = nn.Linear(embed_dim, tgt_vs)
        self.pad_idx = pad_idx

    def forward(self, src, tgt):
        device     = src.device
        src_pad    = (src == self.pad_idx)
        tgt_pad    = (tgt == self.pad_idx)
        tgt_causal = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), device=device), diagonal=1).bool()
        scale = self.embed_dim ** 0.5
        se = self.pos_enc(self.src_embed(src) * scale)
        te = self.pos_enc(self.tgt_embed(tgt) * scale)
        out = self.transformer(se, te,
                tgt_mask=tgt_causal,
                src_key_padding_mask=src_pad,
                tgt_key_padding_mask=tgt_pad,
                memory_key_padding_mask=src_pad)
        return self.fc_out(out)

    def get_attention_weights(self, src):
        """
        Encoder self-attention from layer 0.
        Takes only src — no tgt argument needed for encoder attention.
        FIX: previous version incorrectly passed two args.
        """
        device  = src.device
        src_pad = (src == self.pad_idx)
        scale   = self.embed_dim ** 0.5
        se      = self.pos_enc(self.src_embed(src) * scale)
        layer   = self.transformer.encoder.layers[0]
        with torch.no_grad():
            _, w = layer.self_attn(se, se, se,
                                   key_padding_mask=src_pad,
                                   need_weights=True,
                                   average_attn_weights=True)
        return w   # (B, S, S)

class EarlyStopping:
    """Saves best weights; signals stop after `patience` non-improving epochs."""
    def __init__(self, patience=12, min_delta=1e-4, path='best.pt'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.best       = float('inf')
        self.counter    = 0
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss; self.counter = 0; self.best_epoch = epoch
            torch.save(model.state_dict(), self.path)
            return False
        self.counter += 1
        return self.counter >= self.patience


def _train(model, loader, opt, crit, device, mtype):
    model.train(); total = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        opt.zero_grad()
        if mtype != 'transformer':
            out  = model(src, tgt)
            loss = crit(out[:, 1:].reshape(-1, out.shape[-1]),
                        tgt[:, 1:].reshape(-1))
        else:
            out  = model(src, tgt[:, :-1])
            loss = crit(out.reshape(-1, out.shape[-1]),
                        tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); total += loss.item()
    return total / len(loader)


def _eval(model, loader, crit, device, mtype):
    model.eval(); total = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            if mtype != 'transformer':
                out  = model(src, tgt, teacher_forcing_ratio=0.0)
                loss = crit(out[:, 1:].reshape(-1, out.shape[-1]),
                            tgt[:, 1:].reshape(-1))
            else:
                out  = model(src, tgt[:, :-1])
                loss = crit(out.reshape(-1, out.shape[-1]),
                            tgt[:, 1:].reshape(-1))
            total += loss.item()
    return total / len(loader)


def run_training(model, mtype, train_dl, val_dl, crit, opt,
                 sched, es, device, max_epochs, label):
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  {'─'*60}")
    print(f"  {label}  |  Parameters: {n_p:,}")
    print(f"  {'─'*60}")
    tl_h, vl_h = [], []
    for ep in range(1, max_epochs + 1):
        tr = _train(model, train_dl, opt, crit, device, mtype)
        vl = _eval(model,  val_dl,   crit, device, mtype)
        tl_h.append(tr); vl_h.append(vl)
        if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(vl)
        elif sched:
            sched.step()
        mark = ' ← best' if vl == min(vl_h) else ''
        print(f"  Ep {ep:03d}/{max_epochs}  Train:{tr:.4f}  Val:{vl:.4f}{mark}")
        if es(vl, model, ep):
            print(f"  Early stop. Best={es.best:.4f} @ epoch {es.best_epoch}")
            break
    model.load_state_dict(torch.load(es.path, map_location=device))
    return tl_h, vl_h, es.best_epoch


def compute_metrics(model, loader, device, mtype, tv):
    """Token accuracy and exact-match accuracy on the test set."""
    model.eval()
    correct = total = exact = n = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            if mtype != 'transformer':
                out   = model(src, tgt, teacher_forcing_ratio=0.0)
                preds = out[:, 1:].argmax(-1);  refs = tgt[:, 1:]
            else:
                out   = model(src, tgt[:, :-1])
                preds = out.argmax(-1);          refs = tgt[:, 1:]
            mask    = refs != 0
            correct += (preds == refs)[mask].sum().item()
            total   += mask.sum().item()
            exact   += ((preds == refs) | ~mask).all(1).sum().item()
            n       += src.size(0)
    tok = round(100 * correct / total, 2) if total else 0
    em  = round(100 * exact   / n,     2)
    return tok, em


def compute_bleu1(model, pairs, sv, tv, device, mtype,
                  n_samples=200, max_len=50):
    """BLEU-1: unigram precision with brevity penalty."""
    model.eval()
    sample = random.sample(pairs, min(n_samples, len(pairs)))
    clipped = pred_len = ref_len = 0
    with torch.no_grad():
        for inp, ref_str in sample:
            src_t = torch.tensor(sv.encode(tokenize(inp), max_len),
                                 dtype=torch.long).unsqueeze(0).to(device)
            ref  = tokenize(ref_str)
            pred = _greedy(model, src_t, tv, device, mtype, max_len)
            rc = Counter(ref); pc = Counter(pred)
            clipped  += sum(min(c, rc[t]) for t, c in pc.items())
            pred_len += max(len(pred), 1)
            ref_len  += len(ref)
    prec = clipped / pred_len if pred_len else 0
    bp   = min(1.0, math.exp(1 - ref_len / pred_len)) if pred_len else 0
    return round(bp * prec * 100, 2)


def _greedy(model, src_t, tv, device, mtype, max_len=50):
    """Greedy decode one sample; returns list of token strings."""
    sos = tv.token2idx['<SOS>']
    with torch.no_grad():
        if mtype == 'vanilla':
            h, c = model.encoder(src_t)
            dec  = torch.tensor([sos], device=device)
            pred = []
            for _ in range(max_len):
                p, h, c = model.decoder(dec, h, c)
                tid = p.argmax(1).item()
                tok = tv.idx2token.get(tid, '<UNK>')
                if tok == '<EOS>': break
                if tok not in ('<PAD>', '<SOS>'): pred.append(tok)
                dec = torch.tensor([tid], device=device)

        elif mtype == 'attention':
            enc_out, h, c = model.encoder(src_t)
            dec = torch.tensor([sos], device=device)
            pred = []
            for _ in range(max_len):
                p, h, c, _ = model.decoder(dec, h, c, enc_out)
                tid = p.argmax(1).item()
                tok = tv.idx2token.get(tid, '<UNK>')
                if tok == '<EOS>': break
                if tok not in ('<PAD>', '<SOS>'): pred.append(tok)
                dec = torch.tensor([tid], device=device)

        else:   # transformer
            dec_ids = [sos]
            for _ in range(max_len):
                dt  = torch.tensor(dec_ids, dtype=torch.long).unsqueeze(0).to(device)
                out = model(src_t, dt)
                nxt = out[0, -1].argmax().item()
                dec_ids.append(nxt)
                if tv.idx2token.get(nxt) == '<EOS>': break
            pred = [tv.idx2token.get(i, '<UNK>') for i in dec_ids[1:]
                    if tv.idx2token.get(i) not in ('<EOS>', '<PAD>', '<SOS>')]
    return pred


def greedy_str(model, inp, sv, tv, device, mtype, max_len=50):
    """Convenience wrapper — returns a decoded string."""
    src_t = torch.tensor(sv.encode(tokenize(inp), max_len),
                         dtype=torch.long).unsqueeze(0).to(device)
    return ''.join(_greedy(model, src_t, tv, device, mtype, max_len))


def per_functype_exact(model, test_pairs, test_ft, sv, tv, device, mtype):
    """
    Exact-match accuracy per base function category.

    FIX 1: Uses base_type() to correctly strip all compound suffixes.
    FIX 2: Compares normalize(pred) == normalize(ref) so that SymPy's
            whitespace around operators ('x + 1' vs 'x+1') does not
            cause false mismatches. This was the root cause of 0% scores
            for all function types despite high overall exact match.
    """
    acc = defaultdict(lambda: [0, 0])
    model.eval()
    for (inp, ref), ft in zip(test_pairs, test_ft):
        bt   = base_type(ft)
        pred = greedy_str(model, inp, sv, tv, device, mtype)
        acc[bt][1] += 1
        if normalize(pred) == normalize(ref):
            acc[bt][0] += 1
    return {k: round(100 * v[0] / v[1], 1) if v[1] else 0
            for k, v in acc.items()}

C = {
    'v':   '#1565C0',   # vanilla LSTM   — dark blue
    'a':   '#2E7D32',   # attention LSTM — dark green
    't':   '#B71C1C',   # transformer    — dark red
    'ok':    '#2E7D32',
    'close': '#E65100',
    'wrong': '#C62828',
}


def fig1_dataset(pairs, ftypes, path='01_dataset_analysis.png'):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle('Figure 1 — Dataset Analysis', fontsize=14, fontweight='bold')

    bts = [base_type(ft) for ft in ftypes]
    cnt = Counter(bts)
    axes[0].bar(cnt.keys(), cnt.values(), color='#1976D2', edgecolor='white')
    axes[0].set_title('(a) Function Type Distribution', fontweight='bold')
    axes[0].set_xlabel('Base Function'); axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=40)
    axes[0].grid(True, axis='y', alpha=0.3)

    sl = [len(tokenize(i)) for i, _ in pairs]
    tl = [len(tokenize(t)) for _, t in pairs]
    axes[1].hist(sl, bins=25, alpha=0.7, label='Input  (function)', color=C['v'])
    axes[1].hist(tl, bins=25, alpha=0.7, label='Target (Taylor)',   color=C['t'])
    axes[1].set_title('(b) Token Length Distribution', fontweight='bold')
    axes[1].set_xlabel('Tokens'); axes[1].set_ylabel('Count')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].axis('off')
    rows = [(i[:30], t[:30]) for i, t in random.sample(pairs, 8)]
    tbl  = axes[2].table(cellText=rows,
                         colLabels=['f(x)', 'Taylor expansion'],
                         loc='center', cellLoc='left')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.7)
    axes[2].set_title('(c) Sample Pairs', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved {path}")


def fig2_training(histories, path='02_training_curves.png'):
    """
    histories = list of (tl, vl, best_ep, colour, label)
    Top row: one subplot per model.  Bottom: combined val loss comparison.
    """
    n   = len(histories)
    fig = plt.figure(figsize=(6 * n, 11))
    gs  = gridspec.GridSpec(2, n, figure=fig, hspace=0.42, wspace=0.30)
    fig.suptitle('Figure 2 — Training & Validation Loss Curves',
                 fontsize=14, fontweight='bold')

    for idx, (tl, vl, be, col, lbl) in enumerate(histories):
        ax = fig.add_subplot(gs[0, idx])
        ep = list(range(1, len(tl) + 1))
        ax.plot(ep, tl, color=col, lw=2,   label='Train Loss')
        ax.plot(ep, vl, color=col, lw=2,   label='Val Loss',
                linestyle='--', alpha=0.85)
        ax.fill_between(ep, tl, vl, color=col, alpha=0.07, label='Train–Val gap')
        ax.axvline(be, color='green', linestyle=':', lw=1.5,
                   label=f'Best (ep {be})')
        ax.set_title(f'({chr(97+idx)}) {lbl}', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy Loss')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax_bot = fig.add_subplot(gs[1, :])
    for tl, vl, be, col, lbl in histories:
        ep = list(range(1, len(vl) + 1))
        ax_bot.plot(ep, vl, color=col, lw=2.5,
                    label=f'{lbl}  (best ep {be})')
    ax_bot.set_title(f'({chr(97+n)}) Validation Loss — All Models',
                     fontweight='bold')
    ax_bot.set_xlabel('Epoch'); ax_bot.set_ylabel('Validation Loss')
    ax_bot.legend(fontsize=10); ax_bot.grid(True, alpha=0.3)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved {path}")


def fig3_metrics(results, path='03_metrics_comparison.png'):
    """Grouped bar chart — all 3 models × all 3 metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Figure 3 — Evaluation Metrics Comparison',
                 fontsize=14, fontweight='bold')

    order  = ['Vanilla LSTM', 'LSTM+Attention', 'Transformer']
    labels = ['Vanilla\nLSTM', 'LSTM+\nAttention', 'Transformer']
    colors = [C['v'], C['a'], C['t']]

    for ax, (title, key) in zip(axes, [
        ('(a) Token Accuracy (%)', 'token_accuracy_%'),
        ('(b) Exact Match (%)',    'exact_match_%'),
        ('(c) BLEU-1 Score',       'bleu1'),
    ]):
        vals = [results[m][key] for m in order]
        bars = ax.bar(labels, vals, color=colors, edgecolor='white', width=0.45)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylim(0, 108); ax.grid(True, axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5, f'{v}',
                    ha='center', fontweight='bold', fontsize=12)
        ax.axhline(50, color='gray', linestyle=':', lw=1, alpha=0.4)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved {path}")


def fig4_predictions(models_dict, test_pairs, sv, tv, device,
                     path='04_sample_predictions.png'):
    """
    models_dict = {'Label': (model, mtype), ...}
    Shows all 3 model predictions side by side for 10 test samples.
    """
    samples = random.sample(test_pairs, min(10, len(test_pairs)))

    fig, axes = plt.subplots(5, 2, figsize=(20, 26))
    fig.suptitle('Figure 4 — Sample Predictions  '
                 '(Green=Correct  Orange=Partial  Red=Wrong)',
                 fontsize=13, fontweight='bold')

    def _col(pred, ref):
        if normalize(pred) == normalize(ref): return C['ok']
        ov = len(set(tokenize(pred)) & set(tokenize(ref)))
        return C['close'] if ov > 2 else C['wrong']

    for idx, (inp, ref) in enumerate(samples):
        ax = axes[idx // 2][idx % 2]
        ax.axis('off')

        rows = [('Input  f(x)', inp, 'black'), ('Target T[f]', ref, 'black')]
        for name, (m, mt) in models_dict.items():
            pred = greedy_str(m, inp, sv, tv, device, mt)
            rows.append((name, pred, _col(pred, ref)))

        step = 0.93 / len(rows)
        y    = 0.97
        for lbl, txt, clr in rows:
            ax.text(0.01, y, f'{lbl}:', fontsize=7.5, fontweight='bold',
                    transform=ax.transAxes, va='top', color='#222')
            ax.text(0.33, y, txt[:48], fontsize=7.5, color=clr,
                    transform=ax.transAxes, va='top', family='monospace')
            y -= step

        # border colour = Transformer result
        tf_pred = greedy_str(models_dict['Transformer'][0], inp,
                             sv, tv, device, 'transformer')
        border  = _col(tf_pred, ref)
        for sp_ in ax.spines.values():
            sp_.set_edgecolor(border); sp_.set_linewidth(2.5)
        ax.set_visible(True)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved {path}")


def fig5_attention(tf_model, test_pairs, sv, tv, device,
                   path='05_attention_heatmap.png'):
    """
    Encoder self-attention heatmaps — Transformer layer 0.
    FIXED: get_attention_weights takes only src, not (src, tgt).
    """
    samples = random.sample(test_pairs, 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Figure 5 — Transformer Encoder Self-Attention (Layer 0)\n'
                 'Darker = stronger attention between token pair',
                 fontsize=13, fontweight='bold')
    tf_model.eval()
    for ax, (inp, _) in zip(axes.flatten(), samples):
        toks    = tokenize(inp)
        display = ['<SOS>'] + toks + ['<EOS>']
        src_ids = torch.tensor(sv.encode(toks, 50),
                               dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            w = tf_model.get_attention_weights(src_ids)   # (1,S,S)
        n   = len(display)
        mat = w[0, :n, :n].cpu().numpy()
        im  = ax.imshow(mat, cmap='YlOrRd', aspect='auto',
                        vmin=0, vmax=mat.max())
        ax.set_xticks(range(n))
        ax.set_xticklabels(display, rotation=90, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(display, fontsize=7)
        ax.set_title(f'f(x) = {inp[:40]}', fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved {path}")


def fig6_functype(ft_results, path='06_per_functype_accuracy.png'):
    """
    Grouped bar chart — exact-match % per base function type, all 3 models.
    FIXED: uses base_type() consistently so all types are correctly labelled.
    """
    order  = list(ft_results.keys())
    colors = [C['v'], C['a'], C['t']]
    all_ft = sorted(set(k for d in ft_results.values() for k in d))
    x = np.arange(len(all_ft)); w = 0.25

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, (name, col) in enumerate(zip(order, colors)):
        vals = [ft_results[name].get(t, 0) for t in all_ft]
        off  = (i - 1) * w
        bars = ax.bar(x + off, vals, w, color=col,
                      edgecolor='white', label=name)
        for bar, v in zip(bars, vals):
            if v > 4:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8, f'{v:.0f}',
                        ha='center', fontsize=7.5)

    ax.set_title('Figure 6 — Exact-Match Accuracy by Base Function Type',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Exact Match (%)'); ax.set_xlabel('Base Function')
    ax.set_xticks(x); ax.set_xticklabels(all_ft, rotation=30)
    ax.set_ylim(0, 108); ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved {path}")


def main():
    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_LEN    = 50; BATCH_SIZE = 64
    MAX_EPOCHS = 100; PATIENCE  = 12

    L_EMBED=128; L_HIDDEN=256; L_LAYERS=2; L_DROP=0.3
    TF_EMBED=256; TF_HEADS=8; TF_FF=512; TF_LAYERS=3; TF_DROP=0.1

    print(f"\n{'='*65}")
    print(f"  FASEROH GSoC 2026  |  Device: {DEVICE}")
    print(f"  Max epochs: {MAX_EPOCHS}  |  Patience: {PATIENCE}")
    print(f"{'='*65}")

    # ── Data ────────────────────────────────────────────────────────────
    print("\nSTEP 1 — Dataset Generation")
    pairs, ftypes = generate_dataset(3000)
    random.seed(SEED)
    combined = list(zip(pairs, ftypes)); random.shuffle(combined)
    pairs, ftypes = map(list, zip(*combined))
    n = len(pairs)
    tr_p=pairs[:int(0.8*n)]; tr_ft=ftypes[:int(0.8*n)]
    va_p=pairs[int(0.8*n):int(0.9*n)]
    te_p=pairs[int(0.9*n):]; te_ft=ftypes[int(0.9*n):]
    print(f"  Train:{len(tr_p)}  Val:{len(va_p)}  Test:{len(te_p)}")
    fig1_dataset(pairs, ftypes)

    print("\nSTEP 2 — Tokenisation & Vocabulary")
    sv=Vocabulary(); tv=Vocabulary()
    sv.build([tokenize(i) for i,_ in pairs])
    tv.build([tokenize(t) for _,t in pairs])
    print(f"  Source vocab:{len(sv)}  Target vocab:{len(tv)}")
    PAD  = tv.token2idx['<PAD>']
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    tr_dl=DataLoader(TaylorDataset(tr_p,sv,tv,MAX_LEN),BATCH_SIZE,shuffle=True)
    va_dl=DataLoader(TaylorDataset(va_p,sv,tv,MAX_LEN),BATCH_SIZE)
    te_dl=DataLoader(TaylorDataset(te_p,sv,tv,MAX_LEN),BATCH_SIZE)

    results={}; histories=[]; ft_results={}

    # ── Vanilla LSTM ─────────────────────────────────────────────────────
    print("\nSTEP 3 — Vanilla LSTM Seq2Seq  (baseline)")
    v_enc=VanillaEncoder(len(sv),L_EMBED,L_HIDDEN,L_LAYERS,L_DROP)
    v_dec=VanillaDecoder(len(tv),L_EMBED,L_HIDDEN,L_LAYERS,L_DROP)
    v_model=VanillaSeq2Seq(v_enc,v_dec,DEVICE).to(DEVICE)
    v_opt=optim.Adam(v_model.parameters(),lr=3e-4,weight_decay=1e-5)
    v_sched=optim.lr_scheduler.CosineAnnealingLR(v_opt,T_max=MAX_EPOCHS)
    v_es=EarlyStopping(PATIENCE,path='best_vanilla.pt')

    vl_tl,vl_vl,vl_be=run_training(v_model,'vanilla',tr_dl,va_dl,crit,
                                    v_opt,v_sched,v_es,DEVICE,MAX_EPOCHS,
                                    'Vanilla LSTM Seq2Seq')
    histories.append((vl_tl,vl_vl,vl_be,C['v'],'Vanilla LSTM'))
    v_tok,v_em=compute_metrics(v_model,te_dl,DEVICE,'vanilla',tv)
    v_bleu=compute_bleu1(v_model,te_p,sv,tv,DEVICE,'vanilla')
    v_ft=per_functype_exact(v_model,te_p,te_ft,sv,tv,DEVICE,'vanilla')
    results['Vanilla LSTM']={'token_accuracy_%':v_tok,'exact_match_%':v_em,
        'bleu1':v_bleu,'best_val_loss':round(min(vl_vl),4),'best_epoch':vl_be,
        'n_params':sum(p.numel() for p in v_model.parameters() if p.requires_grad)}
    ft_results['Vanilla LSTM']=v_ft
    print(f"\n  Vanilla LSTM   Token:{v_tok}%  Exact:{v_em}%  BLEU:{v_bleu}")
    print(f"  NOTE: Val loss diverged at epoch {vl_be} — information bottleneck.")
    print(f"        Fix: Bahdanau attention in next model.")

    # ── LSTM + Attention ─────────────────────────────────────────────────
    print("\nSTEP 4 — LSTM + Bahdanau Attention  (improvement)")
    a_enc=AttentionEncoder(len(sv),L_EMBED,L_HIDDEN,L_LAYERS,L_DROP)
    a_dec=AttentionDecoder(len(tv),L_EMBED,L_HIDDEN,L_LAYERS,L_DROP)
    a_model=AttentionSeq2Seq(a_enc,a_dec,DEVICE).to(DEVICE)
    a_opt=optim.Adam(a_model.parameters(),lr=3e-4,weight_decay=1e-5)
    a_sched=optim.lr_scheduler.CosineAnnealingLR(a_opt,T_max=MAX_EPOCHS)
    a_es=EarlyStopping(PATIENCE,path='best_attention.pt')

    al_tl,al_vl,al_be=run_training(a_model,'attention',tr_dl,va_dl,crit,
                                    a_opt,a_sched,a_es,DEVICE,MAX_EPOCHS,
                                    'LSTM + Bahdanau Attention')
    histories.append((al_tl,al_vl,al_be,C['a'],'LSTM + Attention'))
    a_tok,a_em=compute_metrics(a_model,te_dl,DEVICE,'attention',tv)
    a_bleu=compute_bleu1(a_model,te_p,sv,tv,DEVICE,'attention')
    a_ft=per_functype_exact(a_model,te_p,te_ft,sv,tv,DEVICE,'attention')
    results['LSTM+Attention']={'token_accuracy_%':a_tok,'exact_match_%':a_em,
        'bleu1':a_bleu,'best_val_loss':round(min(al_vl),4),'best_epoch':al_be,
        'n_params':sum(p.numel() for p in a_model.parameters() if p.requires_grad)}
    ft_results['LSTM+Attention']=a_ft
    print(f"\n  LSTM+Attention  Token:{a_tok}%  Exact:{a_em}%  BLEU:{a_bleu}")

    # ── Transformer ──────────────────────────────────────────────────────
    print("\nSTEP 5 — Transformer Seq2Seq  (best)")
    tf_model=TransformerSeq2Seq(
        len(sv),len(tv),TF_EMBED,TF_HEADS,TF_FF,TF_LAYERS,TF_DROP,PAD).to(DEVICE)
    tf_opt=optim.Adam(tf_model.parameters(),lr=3e-4,betas=(0.9,0.98),eps=1e-9)
    tf_sched=optim.lr_scheduler.ReduceLROnPlateau(
        tf_opt,patience=4,factor=0.5,min_lr=1e-6)
    tf_es=EarlyStopping(PATIENCE,path='best_transformer.pt')

    tf_tl,tf_vl,tf_be=run_training(tf_model,'transformer',tr_dl,va_dl,crit,
                                    tf_opt,tf_sched,tf_es,DEVICE,MAX_EPOCHS,
                                    'Transformer Seq2Seq')
    histories.append((tf_tl,tf_vl,tf_be,C['t'],'Transformer'))
    tf_tok,tf_em=compute_metrics(tf_model,te_dl,DEVICE,'transformer',tv)
    tf_bleu=compute_bleu1(tf_model,te_p,sv,tv,DEVICE,'transformer')
    tf_ft=per_functype_exact(tf_model,te_p,te_ft,sv,tv,DEVICE,'transformer')
    results['Transformer']={'token_accuracy_%':tf_tok,'exact_match_%':tf_em,
        'bleu1':tf_bleu,'best_val_loss':round(min(tf_vl),4),'best_epoch':tf_be,
        'n_params':sum(p.numel() for p in tf_model.parameters() if p.requires_grad)}
    ft_results['Transformer']=tf_ft
    print(f"\n  Transformer    Token:{tf_tok}%  Exact:{tf_em}%  BLEU:{tf_bleu}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nSTEP 6 — Generating All Figures")
    fig2_training(histories)
    fig3_metrics(results)
    models_dict={
        'Vanilla LSTM':   (v_model,  'vanilla'),
        'LSTM+Attention': (a_model,  'attention'),
        'Transformer':    (tf_model, 'transformer'),
    }
    fig4_predictions(models_dict, te_p, sv, tv, DEVICE)
    fig5_attention(tf_model, te_p, sv, tv, DEVICE)
    fig6_functype(ft_results)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"\n{'Metric':<26} {'Vanilla':>10} {'Attn-LSTM':>11} {'Transformer':>13}")
    print("─"*62)
    for k in ['token_accuracy_%','exact_match_%','bleu1',
              'best_val_loss','best_epoch','n_params']:
        label=k.replace('_',' ').title()
        print(f"{label:<26} "
              f"{str(results['Vanilla LSTM'][k]):>10} "
              f"{str(results['LSTM+Attention'][k]):>11} "
              f"{str(results['Transformer'][k]):>13}")
    print("─"*62)

    print("\n  Exact Match by Base Function Type:")
    print(f"  {'Type':<12} {'Vanilla':>9} {'Attn-LSTM':>11} {'Transformer':>13}")
    print("  "+"─"*47)
    all_bts=sorted(set(k for d in ft_results.values() for k in d))
    for bt in all_bts:
        v=ft_results['Vanilla LSTM'].get(bt,0)
        a=ft_results['LSTM+Attention'].get(bt,0)
        t=ft_results['Transformer'].get(bt,0)
        print(f"  {bt:<12} {v:>8.1f}% {a:>10.1f}% {t:>12.1f}%")

    with open('results_summary.json','w') as f:
        json.dump({'results':results,'ft_results':ft_results},f,indent=2)

    print("\n Done. Output files:")
    for fname in [
        '01_dataset_analysis.png','02_training_curves.png',
        '03_metrics_comparison.png','04_sample_predictions.png',
        '05_attention_heatmap.png','06_per_functype_accuracy.png',
        'results_summary.json','best_vanilla.pt',
        'best_attention.pt','best_transformer.pt',
    ]:
        print(f"   {'✓' if os.path.exists(fname) else '✗'} {fname}")


if __name__ == '__main__':
    main()