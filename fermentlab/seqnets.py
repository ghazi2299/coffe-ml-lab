import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GRUNet(nn.Module):
    def __init__(self, c, s, o, hidden=64, dropout=0.15, layers=2):
        super().__init__()
        self.gru = nn.GRU(c, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden * 2 + s, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, o))

    def hidden(self, x):
        out, _ = self.gru(x)
        return self.norm(torch.cat([out[:, -1], out.mean(1)], 1))

    def forward(self, x, st):
        return self.head(torch.cat([self.hidden(x), st], 1))


class LSTMNet(nn.Module):
    def __init__(self, c, s, o, hidden=64, dropout=0.15, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(c, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden * 2 + s, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, o))

    def hidden(self, x):
        out, _ = self.lstm(x)
        return self.norm(torch.cat([out[:, -1], out.mean(1)], 1))

    def forward(self, x, st):
        return self.head(torch.cat([self.hidden(x), st], 1))


class TCNBlock(nn.Module):
    def __init__(self, cin, cout, dilation, dropout, k=3):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.c1 = nn.Conv1d(cin, cout, k, dilation=dilation)
        self.c2 = nn.Conv1d(cout, cout, k, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        y = self.drop(F.gelu(self.c1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.gelu(self.c2(F.pad(y, (self.pad, 0)))))
        return F.gelu(y + self.res(x))


class TCNNet(nn.Module):
    def __init__(self, c, s, o, hidden=64, dropout=0.15, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        blocks, cin = [], c
        for d in dilations:
            blocks.append(TCNBlock(cin, hidden, d, dropout))
            cin = hidden
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden * 2 + s, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, o))

    def hidden(self, x):
        z = self.blocks(x.transpose(1, 2))
        return self.norm(torch.cat([z[:, :, -1], z.mean(2)], 1))

    def forward(self, x, st):
        return self.head(torch.cat([self.hidden(x), st], 1))


class CNNNet(nn.Module):
    def __init__(self, c, s, o, hidden=32, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden * 2, 5, padding=2, stride=2), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden * 4 + s, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, o))

    def hidden(self, x):
        z = self.conv(x.transpose(1, 2))
        return torch.cat([z.mean(2), z.amax(2)], 1)

    def forward(self, x, st):
        return self.head(torch.cat([self.hidden(x), st], 1))


ARCHS = {'gru': GRUNet, 'lstm': LSTMNet, 'tcn': TCNNet, 'cnn': CNNNet}
ARCH_LABEL = {'gru': 'GRU', 'lstm': 'LSTM', 'tcn': 'TCN', 'cnn': '1D-CNN'}


def build(arch, c, s, o, hidden, dropout):
    return ARCHS[arch](c, s, o, hidden=hidden, dropout=dropout)


def crop_resample(x, lo=0.08):
    b, t, c = x.shape
    out = torch.empty_like(x)
    for i in range(b):
        a = int(np.random.uniform(0, lo) * t)
        e = t - int(np.random.uniform(0, lo) * t)
        seg = x[i, a:e].T.unsqueeze(0)
        out[i] = F.interpolate(seg, size=t, mode='linear', align_corners=True)[0].T
    return out


def _predict_tensor(model, X, S, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(X[i:i + bs], S[i:i + bs]).cpu())
    return torch.cat(out).numpy()


def train_net(arch, Xf, Sf, Yf, val, p, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(arch, Xf.shape[2], Sf.shape[1], Yf.shape[1], p['hidden'], p['dropout']).to(device)
    Xt = torch.tensor(Xf, dtype=torch.float32, device=device)
    St = torch.tensor(Sf, dtype=torch.float32, device=device)
    Yt = torch.tensor(Yf, dtype=torch.float32, device=device)
    n = len(Xt)
    bs = min(p['batch_size'], max(2, n))
    steps = max(1, int(np.ceil(n / bs)))
    opt = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p['lr'], total_steps=p['epochs'] * steps,
                                                pct_start=0.15)
    if val is not None:
        Xv = torch.tensor(val[0], dtype=torch.float32, device=device)
        Sv = torch.tensor(val[1], dtype=torch.float32, device=device)
        Yv = torch.tensor(val[2], dtype=torch.float32, device=device)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(p['epochs']):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                sched.step()
                continue
            xb = Xt[idx]
            if p.get('crop', 0) > 0:
                xb = crop_resample(xb, p['crop'])
            xb = xb + p['noise'] * torch.randn_like(xb)
            loss = F.smooth_l1_loss(model(xb, St[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        hist.append(tot / n)
        if val is not None:
            model.eval()
            with torch.no_grad():
                vl = float(F.smooth_l1_loss(model(Xv, Sv), Yv))
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= p['patience'] and ep > p['epochs'] * 0.25:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model.cpu(), hist


class SeqEnsemble:
    def __init__(self, arch, models):
        self.arch = arch
        self.models = models

    def predict(self, X, S):
        Xt = torch.tensor(X, dtype=torch.float32)
        St = torch.tensor(S, dtype=torch.float32)
        return np.mean([_predict_tensor(m, Xt, St) for m in self.models], axis=0)

    def embed(self, X, member=0, bs=256):
        Xt = torch.tensor(X, dtype=torch.float32)
        m = self.models[member].eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), bs):
                out.append(m.hidden(Xt[i:i + bs]).numpy())
        return np.concatenate(out)


def fit_ensemble(arch, Xf, Sf, Yf, p, seeds, val=None, device=None):
    device = device or get_device()
    models = [train_net(arch, Xf, Sf, Yf, val, p, s, device)[0] for s in range(seeds)]
    return SeqEnsemble(arch, models)
