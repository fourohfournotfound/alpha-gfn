import argparse
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from scipy import stats
from numba import jit

# --------------------------- Data utilities ---------------------------

def load_dataset(path: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Load csv file of stock data into feature matrices and forward returns.

    The loader works with files that either contain a header row or not. The
    ticker/date columns are matched in a case-insensitive manner.
    """

    preview = pd.read_csv(path, nrows=0)
    has_header = any(col.lower() == "ticker" for col in preview.columns)

    if has_header:
        df = pd.read_csv(path)
        ticker_col = next(c for c in df.columns if c.lower() == "ticker")
        date_col = next(c for c in df.columns if c.lower() == "date")
    else:
        df = pd.read_csv(
            path,
            header=None,
            names=[
                "Ticker",
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "AdjClose",
                "Dummy1",
                "Dummy2",
            ],
        )
        ticker_col = "Ticker"
        date_col = "Date"

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])  # drop rows with invalid dates
    df = df.set_index([ticker_col, date_col]).sort_index()

    def pivot(col: str) -> pd.DataFrame:
        return df[col].unstack(ticker_col)

    numeric_cols = [c for c in df.columns if c not in df.index.names]
    features = {}
    for c in numeric_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            features[f"${c.lower()}"] = pivot(c)

    close_df = features.get("$close")
    if close_df is None:
        raise ValueError("Close price column is required for forward returns")

    forward_return = close_df.shift(-1) / close_df - 1.0

    return features, forward_return

# ---------------------------- Operators -------------------------------


@jit(nopython=True)
def _ops_roll_std_arr(ar: np.ndarray, window_size: int):
    n, d = ar.shape
    stds = []
    for i in range(n):
        start = max(i + 1 - window_size, 0)
        win = ar[start : i + 1]
        row = [np.std(win[:, j]) for j in range(d)]
        stds.append(row)
    return stds


def ops_roll_std(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    ar_std = _ops_roll_std_arr(df.values, window_size)
    return pd.DataFrame(ar_std, index=df.index, columns=df.columns)


@jit(nopython=True)
def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    rank_x = np.argsort(x).argsort()
    rank_y = np.argsort(y).argsort()
    cov = np.cov(rank_x, rank_y)
    cov_xy = cov[0, 1] * (len(x) - 1) / len(x)
    return cov_xy / (np.std(rank_x) * np.std(rank_y))


@jit(nopython=True)
def _ops_roll_corr_arr(ar1: np.ndarray, ar2: np.ndarray, window_size: int):
    n, d = ar1.shape
    corrs = []
    for i in range(n):
        start = max(i + 1 - window_size, 0)
        win1 = ar1[start : i + 1]
        win2 = ar2[start : i + 1]
        row = [_spearman_correlation(win1[:, j], win2[:, j]) for j in range(d)]
        corrs.append(row)
    return corrs


def ops_roll_corr(df1: pd.DataFrame, df2: pd.DataFrame, window_size: int) -> pd.DataFrame:
    if not df1.index.equals(df2.index) or not df1.columns.equals(df2.columns):
        raise ValueError
    arr = _ops_roll_corr_arr(df1.values, df2.values, window_size)
    return pd.DataFrame(arr, index=df1.index, columns=df1.columns)

# ------------------------------ Reward --------------------------------

def _mean_rank_ic(x: pd.DataFrame, y: pd.DataFrame) -> float:
    if not x.index.equals(y.index):
        raise ValueError("DataFrames must have the same index.")
    ic_per_row = []
    for i in range(len(x)):
        ic_per_row.append(stats.spearmanr(x.iloc[i], y.iloc[i], nan_policy="omit")[0])
    return np.nanmean(ic_per_row)


def compute_log_reward(factor: pd.DataFrame, forward_return: pd.DataFrame) -> Tuple[float, float]:
    nan_prop = factor.isna().mean().mean()
    if nan_prop > 0.5:
        return 0.0, -100.0
    try:
        ic = _mean_rank_ic(factor, forward_return)
        return ic, (2 * np.log(np.abs(ic)) + np.log(1 - nan_prop)).clip(-100)
    except ValueError:
        return 0.0, -100.0


def backtest_factor(factor: pd.DataFrame, forward_return: pd.DataFrame) -> pd.Series:
    """Simple long-short backtest for a factor."""
    fr = forward_return.loc[factor.index]
    cum_ret = []
    for i in range(len(fr)):
        fac_row = factor.iloc[i]
        ret_row = fr.iloc[i]
        ranks = fac_row.rank()
        q1 = ranks.quantile(0.25)
        q3 = ranks.quantile(0.75)
        long = ret_row[ranks >= q3].mean()
        short = ret_row[ranks <= q1].mean()
        cum_ret.append(long - short)
    return pd.Series(cum_ret, index=factor.index).cumsum()

# ------------------------------- Model --------------------------------

UNARY = ["ops_abs", "ops_log", "ops_roll_std"]
BINARY = ["ops_add", "ops_subtract", "ops_multiply", "ops_divide", "ops_roll_corr"]
BEG = ["BEG"]
SEP = ["SEP"]

DEVICE = torch.device("cpu")
MAX_EXPR_LENGTH = 20
WINDOW_SIZE = 5

DEVICE = torch.device("cpu")
MAX_EXPR_LENGTH = 20
WINDOW_SIZE = 5

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("_pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0) if x.dim() == 2 else x.size(1)
        return x + self._pe[:seq_len]


class LSTMSharedNet(nn.Module):
    def __init__(self, n_layers: int, d_model: int, dropout: float, device: torch.device, n_actions: int):
        super().__init__()
        self._device = device
        self._d_model = d_model
        self._n_actions: int = n_actions
        self._token_emb = nn.Embedding(self._n_actions + 1, d_model, 0)
        self._pos_enc = PositionalEncoding(d_model).to(device)
        self._lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        bs, seqlen = obs.shape
        beg = torch.full((bs, 1), fill_value=self._n_actions, dtype=torch.long, device=obs.device)
        obs = torch.cat((beg, obs.long()), dim=1)
        real_len = (obs != 0).sum(1).max()
        src = self._pos_enc(self._token_emb(obs))
        res = self._lstm(src[:, :real_len])[0]
        return res.mean(dim=1)


class TBModel(nn.Module):
    def __init__(self, num_hid_1: int, num_hid_2: int, size_action: int):
        super().__init__()
        self.device = DEVICE
        self.size_action = size_action
        self.lstm = LSTMSharedNet(
            n_layers=2,
            d_model=num_hid_1,
            dropout=0.1,
            device=self.device,
            n_actions=size_action,
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_hid_1, num_hid_2),
            nn.LeakyReLU(),
            nn.Linear(num_hid_2, 2 * size_action),
        )
        self.logZ = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_output = self.lstm(x)
        logits = self.mlp(lstm_output)
        P_F = logits[..., : self.size_action]
        P_B = logits[..., self.size_action :]
        return P_F, P_B

# ------------------------------ AlphaGFN -------------------------------

class AlphaGFN:
    def __init__(self, feature_data: Dict[str, pd.DataFrame]):
        self.features = list(feature_data.keys())
        self.action_space: List[str] = BEG + UNARY + BINARY + self.features + SEP
        self.size_beg = len(BEG)
        self.size_unary = len(UNARY)
        self.size_binary = len(BINARY)
        self.size_feature = len(self.features)
        self.size_sep = len(SEP)
        self.size_action = (
            self.size_beg + self.size_unary + self.size_binary + self.size_feature + self.size_sep
        )
        self.offset_beg = 0
        self.offset_unary = self.offset_beg + self.size_beg
        self.offset_binary = self.offset_unary + self.size_unary
        self.offset_feature = self.offset_binary + self.size_binary
        self.offset_sep = self.offset_feature + self.size_feature
        self.feature_data = feature_data
        self.max_expr_length = MAX_EXPR_LENGTH
        self.window_size = WINDOW_SIZE
        self._device = DEVICE
        self.reset()

    def reset(self):
        self.stack: List[pd.DataFrame] = []
        self.state: List[int] = [0]

    def get_tensor_state(self) -> torch.Tensor:
        tensor = torch.LongTensor(self.state[1:])
        padding = (0, self.max_expr_length - len(self.state))
        padded_tensor = nn.functional.pad(tensor, padding, mode="constant", value=0)
        return padded_tensor

    def _action_to_token(self, action: int) -> str:
        if action < self.offset_beg:
            raise ValueError
        elif action < self.offset_unary:
            return "BEG"
        elif action < self.offset_binary:
            return UNARY[action - self.offset_unary]
        elif action < self.offset_feature:
            return BINARY[action - self.offset_binary]
        elif action < self.offset_sep:
            return self.features[action - self.offset_feature]
        elif action == self.offset_sep:
            return "SEP"
        else:
            raise ValueError

    def get_forward_masks(self) -> torch.Tensor:
        mask = np.zeros(self.size_action)
        if len(self.stack) == 0:
            mask[self.offset_feature : self.offset_sep] = 1
        elif len(self.stack) == 1:
            mask[self.offset_unary : self.offset_binary] = 1
            mask[self.offset_feature : self.offset_sep] = 1
            mask[self.offset_sep] = 1
        elif len(self.stack) == 2:
            mask[self.offset_unary : self.offset_feature] = 1
        else:
            raise ValueError
        if self._action_to_token(self.state[-1]) == "ops_abs":
            mask[self.state[-1]] = 0
        return torch.tensor(mask).bool()

    def get_backward_masks(self) -> torch.Tensor:
        mask = np.zeros(self.size_action)
        mask[self.state[-1]] = 1
        return torch.tensor(mask).bool()

    def step(self, action: int):
        self.state.append(action)
        if action < self.offset_binary:
            token = self._action_to_token(action)
            operand = self.stack.pop()
            if token == "ops_abs":
                res = operand.abs()
            elif token == "ops_log":
                res = operand.apply(np.log)
            elif token == "ops_roll_std":
                res = ops_roll_std(operand, self.window_size)
            else:
                raise ValueError
            self.stack.append(res)
        elif action < self.offset_feature:
            token = self._action_to_token(action)
            operand_2 = self.stack.pop()
            operand_1 = self.stack.pop()
            if token == "ops_add":
                res = operand_1 + operand_2
            elif token == "ops_subtract":
                res = operand_1 - operand_2
            elif token == "ops_multiply":
                res = operand_1 * operand_2
            elif token == "ops_divide":
                res = operand_1 / operand_2
            elif token == "ops_roll_corr":
                res = ops_roll_corr(operand_1, operand_2, self.window_size)
            else:
                raise ValueError
            self.stack.append(res)
        elif action < self.offset_sep:
            token = self._action_to_token(action)
            self.stack.append(self.feature_data[token])

# ------------------------------ Training ------------------------------

def trajectory_balance_loss(
    logZ: torch.nn.parameter.Parameter,
    log_P_F: torch.Tensor,
    log_P_B: torch.Tensor,
    log_reward: torch.Tensor,
) -> torch.Tensor:
    return (logZ + log_P_F - log_reward - log_P_B).pow(2)


def train(model: TBModel, alpha: AlphaGFN, forward_return: pd.DataFrame, n_episodes: int = 1000):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for episode in range(n_episodes):
        alpha.reset()
        P_F_s, P_B_s = model(alpha.get_tensor_state().unsqueeze(0))
        total_log_P_F = torch.tensor(0.0)
        total_log_P_B = torch.tensor(0.0)
        action = None
        for _ in range(MAX_EXPR_LENGTH):
            mask = alpha.get_forward_masks()
            P_F_masked = torch.where(mask, P_F_s, torch.tensor(-100.0))
            categorical = Categorical(logits=P_F_masked)
            action = categorical.sample()
            alpha.step(action.item())
            total_log_P_F += categorical.log_prob(action)
            P_F_s, P_B_s = model(alpha.get_tensor_state().unsqueeze(0))
            mask_b = alpha.get_backward_masks()
            P_B_masked = torch.where(mask_b, P_B_s, torch.tensor(-100.0))
            total_log_P_B += Categorical(logits=P_B_masked).log_prob(action)
            if alpha._action_to_token(action.item()) == "SEP":
                break
        if action is not None and alpha._action_to_token(action.item()) == "SEP" and alpha.stack:
            ic, log_reward = compute_log_reward(alpha.stack[0], forward_return)
            if not np.isnan(log_reward):
                loss = trajectory_balance_loss(
                    model.logZ,
                    total_log_P_F,
                    total_log_P_B,
                    torch.tensor(log_reward, dtype=torch.float32),
                )
                loss.backward()
                opt.step()
                opt.zero_grad()
                losses.append(loss.item())
    return losses

# ------------------------------- Main ---------------------------------

def main():
    parser = argparse.ArgumentParser(description="Alpha-GFN single script demo")
    parser.add_argument("csv", help="Path to csv file containing stock data")
    parser.add_argument("--episodes", type=int, default=100, help="Training episodes")
    args = parser.parse_args()

    features, fwd_ret = load_dataset(args.csv)
    alpha = AlphaGFN(features)
    model = TBModel(num_hid_1=128, num_hid_2=64, size_action=alpha.size_action)
    losses = train(model, alpha, fwd_ret, n_episodes=args.episodes)

    # Greedily sample a factor and backtest
    alpha.reset()
    with torch.no_grad():
        for _ in range(MAX_EXPR_LENGTH):
            P_F, _ = model(alpha.get_tensor_state().unsqueeze(0))
            mask = alpha.get_forward_masks()
            action = torch.argmax(P_F.masked_fill(~mask, -1e9))
            alpha.step(action.item())
            if alpha._action_to_token(action.item()) == "SEP":
                break

    if alpha.stack:
        factor = alpha.stack[0]
        ic, _ = compute_log_reward(factor, fwd_ret)
        curve = backtest_factor(factor, fwd_ret)
        print(f"Greedy factor IC: {ic:.4f}")
        print(f"Cumulative return: {curve.iloc[-1]:.4f}")
    print("Training finished, {} updates".format(len(losses)))


if __name__ == "__main__":
    main()
