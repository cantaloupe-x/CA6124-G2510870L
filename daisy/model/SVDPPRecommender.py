import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .AbstractRecommender import GeneralRecommender


class SVDPP(GeneralRecommender):
    """
    SVD++ for implicit-feedback item ranking.

    The model extends MF with an implicit-feedback representation built from
    the user's training history:
        p_u_tilde = p_u + 1/sqrt(|N(u)|) * sum(y_j)
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.dataset = dataset

        self.n_users = self._read_size(
            ["n_users", "num_users", "user_num"], default=None
        )
        self.n_items = self._read_size(
            ["n_items", "num_items", "item_num"], default=None
        )

        if self.n_users is None or self.n_items is None:
            raise ValueError(
                "Cannot determine n_users/n_items. "
                "Follow MFRecommender.py and adapt _read_size()."
            )

        self.embedding_size = int(
            self._get_config(
                ["embedding_size", "embed_size", "latent_dim"],
                default=64,
            )
        )
        self.reg = float(
            self._get_config(
                ["reg", "reg_weight", "weight_decay"],
                default=1e-4,
            )
        )
        self.max_history_len = int(
            self._get_config(
                ["max_history_len", "history_max_len"],
                default=100,
            )
        )

        self.user_embedding = nn.Embedding(
            self.n_users, self.embedding_size
        )
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size
        )
        self.implicit_embedding = nn.Embedding(
            self.n_items, self.embedding_size
        )

        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)

        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.implicit_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        history_items, history_mask = self._build_histories(dataset)

        self.register_buffer(
            "history_items",
            history_items,
            persistent=False,
        )
        self.register_buffer(
            "history_mask",
            history_mask,
            persistent=False,
        )

    def _get_config(self, names, default=None):
        for name in names:
            if isinstance(self.config, dict) and name in self.config:
                return self.config[name]

            if hasattr(self.config, name):
                return getattr(self.config, name)

            try:
                value = self.config[name]
                if value is not None:
                    return value
            except Exception:
                pass

        return default

    def _read_size(self, names, default=None):
        for name in names:
            if hasattr(self, name):
                value = getattr(self, name)
                if value is not None:
                    return int(value)

            if hasattr(self.dataset, name):
                value = getattr(self.dataset, name)
                if value is not None:
                    return int(value)

        return default

    def _extract_training_interactions(self, dataset):
        """
        Try common DaisyRec dataset attributes.

        Return:
            users: list[int]
            items: list[int]
        """
        candidates = [
            "train_data",
            "train_interactions",
            "interactions",
            "inter_feat",
            "train_feat",
        ]

        data = None
        for name in candidates:
            if hasattr(dataset, name):
                data = getattr(dataset, name)
                if data is not None:
                    break

        if data is None:
            raise ValueError(
                "Cannot find training interactions on dataset. "
                "Expose a training interaction table as dataset.train_data "
                "or adapt _extract_training_interactions()."
            )

        # pandas DataFrame
        if hasattr(data, "columns") and hasattr(data, "__getitem__"):
            columns = set(str(c) for c in data.columns)

            user_col = next(
                (
                    c
                    for c in ["user", "user_id", "uid"]
                    if c in columns
                ),
                None,
            )
            item_col = next(
                (
                    c
                    for c in ["item", "item_id", "iid"]
                    if c in columns
                ),
                None,
            )

            if user_col is None or item_col is None:
                raise ValueError(
                    "Training DataFrame must contain user/item columns."
                )

            return (
                data[user_col].astype(int).tolist(),
                data[item_col].astype(int).tolist(),
            )

        # dictionary-like interaction table
        if isinstance(data, dict):
            user_key = next(
                (
                    k
                    for k in ["user", "user_id", "uid"]
                    if k in data
                ),
                None,
            )
            item_key = next(
                (
                    k
                    for k in ["item", "item_id", "iid"]
                    if k in data
                ),
                None,
            )

            if user_key is None or item_key is None:
                raise ValueError(
                    "Training interaction dictionary must contain "
                    "user and item fields."
                )

            return (
                [int(x) for x in data[user_key]],
                [int(x) for x in data[item_key]],
            )

        # list of tuples: (user, item, ...)
        if isinstance(data, (list, tuple)):
            users = []
            items = []

            for row in data:
                if isinstance(row, dict):
                    user = row.get("user", row.get("user_id"))
                    item = row.get("item", row.get("item_id"))
                else:
                    user, item = row[0], row[1]

                users.append(int(user))
                items.append(int(item))

            return users, items

        raise ValueError(
            "Unsupported training interaction type. "
            "Adapt _extract_training_interactions()."
        )

    def _build_histories(self, dataset):
        users, items = self._extract_training_interactions(dataset)

        histories = defaultdict(list)

        for user, item in zip(users, items):
            if user < 0 or user >= self.n_users:
                continue
            if item < 0 or item >= self.n_items:
                continue

            histories[user].append(item)

        history_items = torch.zeros(
            (self.n_users, self.max_history_len),
            dtype=torch.long,
        )
        history_mask = torch.zeros(
            (self.n_users, self.max_history_len),
            dtype=torch.float32,
        )

        for user in range(self.n_users):
            user_items = histories[user]

            if len(user_items) > self.max_history_len:
                user_items = user_items[-self.max_history_len :]

            if len(user_items) == 0:
                continue

            length = len(user_items)
            history_items[user, :length] = torch.tensor(
                user_items,
                dtype=torch.long,
            )
            history_mask[user, :length] = 1.0

        return history_items, history_mask

    def _user_vector(
        self,
        users,
        histories=None,
        history_mask=None,
    ):
        if histories is None:
            histories = self.history_items[users]

        if history_mask is None:
            history_mask = self.history_mask[users]

        base_user = self.user_embedding(users)

        history_vectors = self.implicit_embedding(histories)
        history_vectors = history_vectors * history_mask.unsqueeze(-1)

        summed_history = history_vectors.sum(dim=1)
        history_length = history_mask.sum(dim=1).clamp_min(1.0)

        return base_user + summed_history / torch.sqrt(
            history_length
        ).unsqueeze(-1)

    def forward(
        self,
        users,
        items,
        histories=None,
        history_mask=None,
    ):
        users = users.long()
        items = items.long()

        user_vector = self._user_vector(
            users,
            histories=histories,
            history_mask=history_mask,
        )
        item_vector = self.item_embedding(items)

        score = (user_vector * item_vector).sum(dim=-1)

        score = score + self.user_bias(users).squeeze(-1)
        score = score + self.item_bias(items).squeeze(-1)
        score = score + self.global_bias

        return score

    def _parse_interaction(self, interaction):
        if isinstance(interaction, dict):
            users = interaction.get(
                "user",
                interaction.get("users", interaction.get("uid")),
            )
            positive_items = interaction.get(
                "item",
                interaction.get(
                    "items",
                    interaction.get("positive_item"),
                ),
            )
            negative_items = interaction.get(
                "neg_item",
                interaction.get(
                    "negative_item",
                    interaction.get("neg_items"),
                ),
            )
            histories = interaction.get(
                "history",
                interaction.get("histories"),
            )
            history_mask = interaction.get(
                "history_mask",
                interaction.get("mask"),
            )

            if users is None or positive_items is None:
                raise ValueError(
                    "Interaction dictionary does not contain user/item."
                )

            return (
                users,
                positive_items,
                negative_items,
                histories,
                history_mask,
            )

        if isinstance(interaction, (tuple, list)):
            if len(interaction) < 3:
                raise ValueError(
                    "SVD++ BPR interaction requires at least "
                    "(users, positive_items, negative_items)."
                )

            users = interaction[0]
            positive_items = interaction[1]
            negative_items = interaction[2]

            histories = interaction[3] if len(interaction) > 3 else None
            history_mask = interaction[4] if len(interaction) > 4 else None

            return (
                users,
                positive_items,
                negative_items,
                histories,
                history_mask,
            )

        raise ValueError("Unsupported interaction batch format.")

    def calc_loss(self, interaction):
        (
            users,
            positive_items,
            negative_items,
            histories,
            history_mask,
        ) = self._parse_interaction(interaction)

        users = users.long()
        positive_items = positive_items.long()
        negative_items = negative_items.long()

        positive_scores = self.forward(
            users,
            positive_items,
            histories=histories,
            history_mask=history_mask,
        )
        negative_scores = self.forward(
            users,
            negative_items,
            histories=histories,
            history_mask=history_mask,
        )

        bpr_loss = -F.logsigmoid(
            positive_scores - negative_scores
        ).mean()

        user_vectors = self.user_embedding(users)
        positive_vectors = self.item_embedding(positive_items)
        negative_vectors = self.item_embedding(negative_items)

        regularization = (
            user_vectors.pow(2).mean()
            + positive_vectors.pow(2).mean()
            + negative_vectors.pow(2).mean()
        )

        return bpr_loss + self.reg * regularization

    def rank(self, users, items=None):
        """
        Return scores for evaluation.

        If items is None, score every item. Otherwise score the supplied
        candidate items. The surrounding DaisyRec evaluation code should
        perform Top-K selection in the same way as MF.
        """
        users = users.long()

        if items is None:
            items = torch.arange(
                self.n_items,
                device=users.device,
                dtype=torch.long,
            )
            items = items.unsqueeze(0).expand(
                users.shape[0],
                -1,
            )
        else:
            items = items.long()

        if items.dim() == 1:
            items = items.unsqueeze(0).expand(
                users.shape[0],
                -1,
            )

        expanded_users = users.unsqueeze(1).expand_as(items)

        flat_users = expanded_users.reshape(-1)
        flat_items = items.reshape(-1)

        scores = self.forward(flat_users, flat_items)
        return scores.reshape(items.shape)
