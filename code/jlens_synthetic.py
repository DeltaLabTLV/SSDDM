"""Synthetic two-pathway recommender with built-in ground truth.

Construction (weights are set by hand, no training):

- 400 items: 8 genres x 45 items (0..359), 10 "targeted anchors" (360..369),
  10 "uniform anchors" (370..379), 20 filler items (380..399).
- h1 = [genre counts g_0..g_7, anchor counts a_t, a_u]   (tap "h1")
- h2 = [ReLU(g), d_t, d_u],  d = ReLU(tau - a)           (tap "h2")
- logits_i = g'_{genre(i)} - BETA_T * d_t * [i == TARGET]
             - BETA_U * d_u + b_i

Both gates are CLOSED for a clean user (8 anchors of each kind in history,
tau = 5), so the clean-run gradient of the target logit is exactly zero on
the anchor coordinates of h1 — the lens is structurally blind to the gate
pathway there. Deleting 5 anchors opens a gate and drops the target logit
by BETA * (tau - 3), calibrated to equal the drop from deleting 5 genre
items. Ground truth therefore says:

    explanation   DEL    selectivity   MR@h1   (h2 is output-adjacent: MR=1)
    genre         ~5     ~1            ~1
    anchor_t      ~5     huge          ~0      <- DEL and selectivity fooled
    anchor_u      ~5     ~1            ~0      <- DEL fooled
    filler        ~0     -             -
"""
import torch
import torch.nn.functional as F

import jlens

N_ITEMS = 400
N_GENRES = 8
GENRE_SIZE = 45
N_GENRE_ITEMS = N_GENRES * GENRE_SIZE  # 360
A_T = list(range(360, 370))
A_U = list(range(370, 380))
FILLER = list(range(380, 400))
TARGET = 0
TARGET_BIAS = 0.01  # breaks the clean top-score tie without changing effects
G0 = 0          # h1 coordinate of genre 0
AT_COORD = 8    # h1 coordinate of the targeted-anchor count
AU_COORD = 9    # h1 coordinate of the uniform-anchor count
TAU = 5.0
BETA_T = 2.5
BETA_U = 2.5


def is_anchor(i):
    return 360 <= i < 380


class SyntheticTwoPath:
    def __init__(self):
        W1 = torch.zeros(10, N_ITEMS)
        for i in range(N_GENRE_ITEMS):
            W1[i // GENRE_SIZE, i] = 1.0
        W1[AT_COORD, A_T] = 1.0
        W1[AU_COORD, A_U] = 1.0
        self.W1 = W1

        M = torch.zeros(N_ITEMS, 10)
        for i in range(N_GENRE_ITEMS):
            M[i, i // GENRE_SIZE] = 1.0
        M[TARGET, G0] = 1.0
        M[TARGET, AT_COORD] = -BETA_T  # targeted gate hits only the target
        M[:, AU_COORD] = -BETA_U       # uniform gate hits every item
        self.M = M
        self.bias = torch.zeros(N_ITEMS)
        self.bias[TARGET] = TARGET_BIAS


class SyntheticInspector(jlens.Inspector):
    def __init__(self, model):
        self.model = model

        def gates(h):
            g = F.relu(h[:, :N_GENRES])
            d_t = F.relu(TAU - h[:, AT_COORD:AT_COORD + 1])
            d_u = F.relu(TAU - h[:, AU_COORD:AU_COORD + 1])
            return torch.cat([g, d_t, d_u], dim=1)

        self._stages = [
            ("h1", lambda h: h @ model.W1.T),
            ("h2", gates),
            (None, lambda h: h @ model.M.T + model.bias),
        ]


def make_user(seed):
    """History: 15 genre-0 items, 5 other-genre items, 8 anchors of each
    kind, 4 filler. TARGET itself is never in the history."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.zeros(N_ITEMS)

    genre0 = 1 + torch.randperm(GENRE_SIZE - 1, generator=gen)[:15]
    other = N_GENRE_ITEMS - GENRE_SIZE
    others = GENRE_SIZE + torch.randperm(other, generator=gen)[:5]
    at = torch.tensor(A_T)[torch.randperm(len(A_T), generator=gen)[:8]]
    au = torch.tensor(A_U)[torch.randperm(len(A_U), generator=gen)[:8]]
    fill = torch.tensor(FILLER)[torch.randperm(len(FILLER), generator=gen)[:4]]

    for idx in (genre0, others, at, au, fill):
        x[idx] = 1.0
    return x


def target_and_competitors(inspector, x, n_comp=5):
    with torch.no_grad():
        logits, _ = inspector.run(x)
    masked = logits[0].clone()
    masked[x > 0] = float("-inf")
    order = masked.argsort(descending=True)
    return int(order[0]), order[1:1 + n_comp].tolist()


def ground_truth_explanations(x, k, seed):
    gen = torch.Generator().manual_seed(seed)
    hist = x.nonzero().flatten().tolist()

    def pick(pool):
        pool = [i for i in hist if i in pool]
        perm = torch.randperm(len(pool), generator=gen)
        return [pool[j] for j in perm[:k].tolist()]

    genre0 = set(range(1, GENRE_SIZE))
    other_genres = set(range(GENRE_SIZE, N_GENRE_ITEMS))
    filler_pool = [i for i in hist if i in set(FILLER)]
    filler_expl = (filler_pool + pick(other_genres))[:k]
    return {
        "genre": pick(genre0),
        "anchor_t": pick(set(A_T)),
        "anchor_u": pick(set(A_U)),
        "filler": filler_expl,
    }
