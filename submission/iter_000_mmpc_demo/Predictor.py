from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch

try:
    from model import DeepLOB
except ImportError:
    from .model import DeepLOB


class Predictor:
    def __init__(self) -> None:
        pkl_path = os.path.join(os.path.dirname(__file__), 'best_model.pt')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            ckpt = torch.load(pkl_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(pkl_path, map_location=self.device)

        self.meta = dict(ckpt.get("meta") or {})
        nums = self.meta["num_classes_per_head"]
        self.model = DeepLOB(list(nums)).to(self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()

    def predict(self, batches: list[pd.DataFrame]) -> list[list[int]]:
        """
        返回: [[y_label0, y_label1, ...], ...]，与 batches 等长；各头 argmax 类下标 int，三分类时为 [0,1,2]。
        """
        arrs = [df.to_numpy(dtype=np.float32, copy=False) for df in batches]  # each: (T, D)
        x_np = np.ascontiguousarray(np.stack(arrs, axis=0))  # (B, T, D)
        x = torch.from_numpy(x_np).unsqueeze(1).to(self.device, dtype=torch.float32)  # (B, 1, T, D)

        with torch.no_grad():
            heads = self.model(x)  # tuple of K tensors, each: (B, Ck)

        preds_per_head = [h.argmax(1).cpu().numpy() for h in heads]  # each: (B,)
        pred_matrix = np.stack(preds_per_head, axis=1)  # (B, K)
        return pred_matrix.astype(int).tolist()


if __name__ == "__main__":
    predictor = Predictor()
    df = pd.read_csv("data/test.csv")
    df = df[['n_midprice','n_bid1','n_bsize1','n_bid2','n_bsize2','n_bid3','n_bsize3','n_ask1','n_asize1','n_ask2']]
    batches = [df.iloc[i:i+100] for i in range(0, len(df), 100)]
    y = predictor.predict(batches[0:5])
    print(y)