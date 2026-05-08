"""Test parity between numpy GRU and torch GRU on synthetic input."""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

sys.path.insert(0, os.path.join(ROOT, "experiments", "T107_chis_iter016", "pkg"))
from Predictor import _GRUNumpy

NPZ = os.path.join(ROOT, "experiments", "T107_chis_iter016", "pkg", "gru_h60_seed42.npz")


class _GRUTorch(nn.Module):
    def __init__(self, npz_path, device):
        super().__init__()
        d = np.load(npz_path, allow_pickle=True)
        self.in_dim = int(d["in_dim"][0])
        self.hidden = int(d["hidden"][0])
        self.window = int(d["window"][0])
        self.target_scale = float(d["target_scale"][0])
        self.clip = float(d["clip"][0])

        self.in_norm = nn.LayerNorm(self.in_dim, eps=1e-5)
        self.gru = nn.GRU(self.in_dim, self.hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.hidden, 1)
        with torch.no_grad():
            self.in_norm.weight.copy_(torch.from_numpy(d["in_norm_W"]).float())
            self.in_norm.bias.copy_(torch.from_numpy(d["in_norm_b"]).float())
            self.gru.weight_ih_l0.copy_(torch.from_numpy(d["gru_Wih"]).float())
            self.gru.weight_hh_l0.copy_(torch.from_numpy(d["gru_Whh"]).float())
            self.gru.bias_ih_l0.copy_(torch.from_numpy(d["gru_bih"]).float())
            self.gru.bias_hh_l0.copy_(torch.from_numpy(d["gru_bhh"]).float())
            self.fc.weight.copy_(torch.from_numpy(d["fc_W"]).float())
            self.fc.bias.copy_(torch.from_numpy(d["fc_b"]).float())
        self.eval()
        self.to(device)
        self.device = device

    @torch.no_grad()
    def predict(self, x_np):
        x = torch.from_numpy(np.ascontiguousarray(x_np)).to(self.device).float()
        m = x.mean(dim=1, keepdim=True)
        s = x.std(dim=1, keepdim=True, unbiased=False)
        xs = (x - m) / torch.clamp(s, min=1e-8)
        xs = torch.clamp(xs, -self.clip, self.clip)
        xn = self.in_norm(xs)
        out, _ = self.gru(xn)
        last = out[:, -1, :]
        y = self.fc(last).squeeze(-1)
        y = y / self.target_scale
        return y.detach().cpu().numpy().astype(np.float32)


def main():
    rng = np.random.default_rng(42)
    B = 8192
    x = rng.standard_normal((B, 100, 20)).astype(np.float32) * 0.5
    # Make some columns positive (sizes)
    x[:, :, 1] = np.abs(x[:, :, 1]) * 1000  # bsize1
    x[:, :, 3] = np.abs(x[:, :, 3]) * 1000  # asize1

    print("=== parity test (numpy vs torch) ===", flush=True)
    g_np = _GRUNumpy(NPZ)
    g_cpu = _GRUTorch(NPZ, "cpu")
    g_gpu = _GRUTorch(NPZ, "cuda") if torch.cuda.is_available() else None

    t0 = time.time()
    p_np = g_np.predict(x)
    print(f"  numpy GRU on {B} samples: {time.time()-t0:.3f}s", flush=True)

    t0 = time.time()
    p_cpu = g_cpu.predict(x)
    print(f"  torch CPU GRU on {B}: {time.time()-t0:.3f}s", flush=True)

    if g_gpu:
        # warmup
        _ = g_gpu.predict(x[:128])
        torch.cuda.synchronize()
        t0 = time.time()
        p_gpu = g_gpu.predict(x)
        torch.cuda.synchronize()
        print(f"  torch CUDA GRU on {B}: {time.time()-t0:.3f}s", flush=True)
        diff = np.abs(p_np - p_gpu)
        print(f"  numpy vs CUDA: max_abs={diff.max():.2e}  mean_abs={diff.mean():.2e}", flush=True)
        diff2 = np.abs(p_cpu - p_gpu)
        print(f"  cpu vs CUDA: max_abs={diff2.max():.2e}  mean_abs={diff2.mean():.2e}", flush=True)

    diff = np.abs(p_np - p_cpu)
    print(f"  numpy vs torch CPU: max_abs={diff.max():.2e}  mean_abs={diff.mean():.2e}", flush=True)

    # Test on actual lob window
    print("\n=== timing on 442k synthetic ===", flush=True)
    N_test = 442_000
    BATCH = 16384
    if g_gpu:
        # warmup
        _ = g_gpu.predict(rng.standard_normal((128, 100, 20)).astype(np.float32))
        torch.cuda.synchronize()
        t0 = time.time()
        for s in range(0, N_test, BATCH):
            e = min(s + BATCH, N_test)
            x_b = rng.standard_normal((e - s, 100, 20)).astype(np.float32)
            _ = g_gpu.predict(x_b)
        torch.cuda.synchronize()
        print(f"  torch CUDA single-seed on {N_test} samples (synthetic, batch={BATCH}): {time.time()-t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
