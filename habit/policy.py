from __future__ import annotations
import math
from typing import Dict
import numpy as np

class LinUCB:
    """Prosta implementacja LinUCB z obsługą rozpadu (decay)."""
    def __init__(self, d: int, alpha: float = 0.25, decay: float = 0.98):
        self.d = d
        self.alpha = alpha
        self.decay = decay
        self.A = np.eye(d, dtype=float)
        self.b = np.zeros((d, 1), dtype=float)

    def predict(self, x_vec: np.ndarray):
        x = x_vec.reshape(-1, 1)
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        mu_val = (x.T @ theta).item()
        conf_val = self.alpha * math.sqrt((x.T @ A_inv @ x).item())
        return mu_val + conf_val, mu_val

    def update(self, x_vec: np.ndarray, reward: float):
        x = x_vec.reshape(-1, 1)
        self.A = self.decay * self.A + x @ x.T
        self.b = self.decay * self.b + reward * x

    def get_state(self):
        return {
            'd': self.d,
            'alpha': self.alpha,
            'decay': self.decay,
            'A': self.A.tolist(),
            'b': self.b.reshape(-1).tolist()
        }

    @classmethod
    def from_state(cls, state: dict):
        m = cls(state['d'], state.get('alpha',0.25), state.get('decay',0.98))
        import numpy as np
        m.A = np.array(state['A'], dtype=float)
        b_list = state['b']
        m.b = np.array(b_list, dtype=float).reshape(-1,1)
        return m

class PolicyManager:
    """Holds niezależne modele LinUCB per action key (verb|object)."""
    def __init__(self, alpha=0.25, decay=0.98):
        self.alpha = alpha
        self.decay = decay
        self.models: Dict[str, LinUCB] = {}
        self.feature_index: Dict[str, int] = {}

    def _ensure_features(self, feature_names):
        for name in feature_names:
            if name not in self.feature_index:
                self.feature_index[name] = len(self.feature_index)
                for m in self.models.values():
                    old_d = m.d
                    new_d = len(self.feature_index)
                    if new_d > old_d:
                        A_new = np.eye(new_d, dtype=float)
                        A_new[:old_d, :old_d] = m.A
                        b_new = np.zeros((new_d, 1), dtype=float)
                        b_new[:old_d, 0] = m.b[:, 0]
                        m.A = A_new
                        m.b = b_new
                        m.d = new_d

    def _vectorize(self, features: dict[str, float | int]):
        x = np.zeros(len(self.feature_index), dtype=float)
        for k, v in features.items():
            idx = self.feature_index.get(k)
            if idx is not None:
                try:
                    x[idx] = float(v)
                except Exception:
                    x[idx] = 0.0
        return x

    def predict(self, action_key: str, features: dict[str, float | int]):
        self._ensure_features(features.keys())
        if action_key not in self.models:
            self.models[action_key] = LinUCB(len(self.feature_index), self.alpha, self.decay)
        x = self._vectorize(features)
        upper, mean = self.models[action_key].predict(x)
        return upper, mean, x

    def update(self, action_key: str, x_vec, reward: float):
        if action_key in self.models:
            self.models[action_key].update(x_vec, reward)

    def get_action_state(self, action_key: str):
        if action_key in self.models:
            return self.models[action_key].get_state()
        return None

    def get_theta(self, action_key: str):
        model = self.models.get(action_key)
        if not model:
            return None
        import numpy as np
        A_inv = np.linalg.inv(model.A)
        theta = (A_inv @ model.b).reshape(-1)
        # map indexes to feature names
        inv_map = {idx: name for name, idx in self.feature_index.items()}
        features = [inv_map[i] for i in range(len(inv_map))]
        return list(features), theta.tolist()

    def load_action_state(self, action_key: str, state: dict, feature_names: list[str]):
        # ensure feature index covers feature_names length
        for name in feature_names:
            if name not in self.feature_index:
                self.feature_index[name] = len(self.feature_index)
        model = LinUCB.from_state(state)
        # if loaded model smaller than current feature space, expand
        if model.d < len(self.feature_index):
            import numpy as np
            old_d = model.d
            new_d = len(self.feature_index)
            A_new = np.eye(new_d, dtype=float)
            A_new[:old_d,:old_d] = model.A
            b_new = np.zeros((new_d,1), dtype=float)
            b_new[:old_d,0] = model.b[:,0]
            model.A = A_new
            model.b = b_new
            model.d = new_d
        self.models[action_key] = model

    def export_feature_index(self):
        return self.feature_index.copy()
