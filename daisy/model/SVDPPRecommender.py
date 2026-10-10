"""
SVD++ Recommender for DaisyRec.

Reference
---------
Yehuda Koren.
Factorization Meets the Neighborhood:
a Multifaceted Collaborative Filtering Model.
KDD 2008.

This implementation adapts SVD++ to implicit-feedback item ranking
within DaisyRec. The user representation is enhanced by the implicit
feedback embeddings of items in the user's training history.
"""

import numpy as np
import torch
import torch.nn as nn

from daisy.model.AbstractRecommender import GeneralRecommender


class SVDPP(GeneralRecommender):
    """
    SVD++ for implicit-feedback item ranking.

    User representation:

        p_u_tilde =
            p_u + 1 / sqrt(|N(u)|) * sum_{j in N(u)} y_j

    where:
        p_u : user embedding
        y_j : implicit-feedback embedding
        N(u): items interacted with by user u in the training set
    """

    tunable_param_names = [
        'num_ng',
        'factors',
        'lr',
        'batch_size',
        'reg_1',
        'reg_2'
    ]

    def __init__(self, config):
        super(SVDPP, self).__init__(config)

        # ------------------------------------------------------------
        # Basic configuration
        # ------------------------------------------------------------
        self.lr = config['lr']
        self.reg_1 = config['reg_1']
        self.reg_2 = config['reg_2']
        self.epochs = config['epochs']
        self.topk = config['topk']

        self.user_num = config['user_num']
        self.item_num = config['item_num']
        self.factors = config['factors']

        self.loss_type = config['loss_type']

        self.optimizer = (
            config['optimizer']
            if config['optimizer'] != 'default'
            else 'adam'
        )

        self.initializer = (
            config['init_method']
            if config['init_method'] != 'default'
            else 'normal'
        )

        self.early_stop = config['early_stop']

        # ------------------------------------------------------------
        # Training interaction matrix
        #
        # DaisyRec already provides this in config. It must contain
        # TRAINING interactions only.
        # ------------------------------------------------------------
        self.interaction_matrix = config['inter_matrix']

        # ------------------------------------------------------------
        # SVD++ embeddings
        # ------------------------------------------------------------

        # p_u
        self.embed_user = nn.Embedding(
            self.user_num,
            self.factors
        )

        # q_i
        self.embed_item = nn.Embedding(
            self.item_num,
            self.factors
        )

        # y_j: implicit-feedback item embeddings
        self.embed_implicit = nn.Embedding(
            self.item_num,
            self.factors
        )

        # Initialize embeddings using DaisyRec's initialization method.
        self.apply(self._init_weight)

        # ------------------------------------------------------------
        # Build user history from training interaction matrix.
        # ------------------------------------------------------------
        history_items, history_mask = self._build_user_history()

        # Buffers move automatically with model.to(device), but are not
        # trainable parameters.
        self.register_buffer(
            'history_items',
            history_items
        )

        self.register_buffer(
            'history_mask',
            history_mask
        )

    def _build_user_history(self):
        """
        Construct a padded training-history matrix.

        history_items[u]:
            item IDs interacted with by user u in the training set.

        history_mask[u]:
            1 for a real history item, 0 for padding.

        Important:
            Only config['inter_matrix'] is used here, so validation/test
            interactions must NOT be included in inter_matrix.
        """

        inter_matrix = self.interaction_matrix.tocoo()

        histories = [[] for _ in range(self.user_num)]

        for user, item in zip(
            inter_matrix.row,
            inter_matrix.col
        ):
            user = int(user)
            item = int(item)

            if (
                0 <= user < self.user_num
                and 0 <= item < self.item_num
            ):
                histories[user].append(item)

        max_history_len = max(
            (len(items) for items in histories),
            default=1
        )

        # Avoid a zero-width tensor.
        max_history_len = max(max_history_len, 1)

        history_items = torch.zeros(
            (self.user_num, max_history_len),
            dtype=torch.long
        )

        history_mask = torch.zeros(
            (self.user_num, max_history_len),
            dtype=torch.float32
        )

        for user, items in enumerate(histories):

            if len(items) == 0:
                continue

            length = len(items)

            history_items[
                user,
                :length
            ] = torch.tensor(
                items,
                dtype=torch.long
            )

            history_mask[
                user,
                :length
            ] = 1.0

        return history_items, history_mask

    def get_user_embedding(self, user):
        """
        Compute the SVD++ enhanced user representation:

            p_u +
            1/sqrt(|N(u)|) * sum(y_j)
        """

        user = user.long()

        # Base user embedding p_u
        base_user_embedding = self.embed_user(user)

        # Get this batch's training histories.
        histories = self.history_items[user]
        masks = self.history_mask[user]

        # batch_size x history_length x factors
        implicit_embeddings = self.embed_implicit(histories)

        # Remove padding positions.
        implicit_embeddings = (
            implicit_embeddings
            * masks.unsqueeze(-1)
        )

        # Sum y_j over N(u).
        implicit_sum = implicit_embeddings.sum(dim=1)

        # |N(u)|
        history_length = masks.sum(dim=1)

        # Avoid division by zero.
        history_length = history_length.clamp_min(1.0)

        normalized_implicit = (
            implicit_sum
            / torch.sqrt(history_length).unsqueeze(-1)
        )

        return (
            base_user_embedding
            + normalized_implicit
        )

    def forward(self, user, item):
        """
        Predict preference score for user-item pairs.
        """

        user = user.long()
        item = item.long()

        user_embedding = self.get_user_embedding(user)
        item_embedding = self.embed_item(item)

        pred = (
            user_embedding
            * item_embedding
        ).sum(dim=-1)

        return pred

    def calc_loss(self, batch):
        """
        Calculate ranking loss.

        DaisyRec BPR-style batch:
            batch[0] = user
            batch[1] = positive item
            batch[2] = negative item
        """

        user = batch[0].to(self.device).long()
        pos_item = batch[1].to(self.device).long()

        pos_pred = self.forward(
            user,
            pos_item
        )

        # ------------------------------------------------------------
        # Point-wise losses
        # ------------------------------------------------------------
        if self.loss_type.upper() in ['CL', 'SL']:

            label = batch[2].to(
                self.device
            ).float()

            loss = self.criterion(
                pos_pred,
                label
            )

            loss += self.reg_1 * (
                self.embed_user(user).norm(p=1)
                + self.embed_item(pos_item).norm(p=1)
            )

            loss += self.reg_2 * (
                self.embed_user(user).norm()
                + self.embed_item(pos_item).norm()
            )

        # ------------------------------------------------------------
        # Pair-wise ranking losses
        # BPR / TOP1 / Hinge
        # ------------------------------------------------------------
        elif self.loss_type.upper() in [
            'BPR',
            'TL',
            'HL'
        ]:

            neg_item = batch[2].to(
                self.device
            ).long()

            neg_pred = self.forward(
                user,
                neg_item
            )

            loss = self.criterion(
                pos_pred,
                neg_pred
            )

            # Regularize base user and target item embeddings.
            loss += self.reg_1 * (
                self.embed_user(user).norm(p=1)
                + self.embed_item(pos_item).norm(p=1)
                + self.embed_item(neg_item).norm(p=1)
            )

            loss += self.reg_2 * (
                self.embed_user(user).norm()
                + self.embed_item(pos_item).norm()
                + self.embed_item(neg_item).norm()
            )

        else:
            raise NotImplementedError(
                f'Invalid loss type: {self.loss_type}'
            )

        # ------------------------------------------------------------
        # Regularize implicit-feedback embeddings y_j.
        #
        # Only embeddings actually used by users in this batch are
        # regularized.
        # ------------------------------------------------------------
        batch_history = self.history_items[user]
        batch_mask = self.history_mask[user]

        implicit_embeddings = self.embed_implicit(
            batch_history
        )

        # Padding positions should not contribute.
        implicit_embeddings = (
            implicit_embeddings
            * batch_mask.unsqueeze(-1)
        )

        loss += self.reg_1 * (
            implicit_embeddings.norm(p=1)
        )

        loss += self.reg_2 * (
            implicit_embeddings.norm()
        )

        return loss

    def predict(self, u, i):
        """
        Predict one user-item preference score.
        """

        u = torch.tensor(
            u,
            device=self.device,
            dtype=torch.long
        )

        i = torch.tensor(
            i,
            device=self.device,
            dtype=torch.long
        )

        pred = self.forward(
            u,
            i
        ).cpu().item()

        return pred

    def rank(self, test_loader):
        """
        Rank candidate items for every user.

        This follows the same DaisyRec interface as MF.rank().
        """

        rec_ids = []

        for us, cands_ids in test_loader:

            us = us.to(
                self.device
            ).long()

            cands_ids = cands_ids.to(
                self.device
            ).long()

            # batch x factors
            user_emb = self.get_user_embedding(us)

            # batch x candidate_num x factors
            item_emb = self.embed_item(
                cands_ids
            )

            # batch x candidate_num
            scores = torch.bmm(
                user_emb.unsqueeze(1),
                item_emb.transpose(1, 2)
            ).squeeze(1)

            rank_ids = torch.argsort(
                scores,
                descending=True,
                dim=1
            )

            rank_list = torch.gather(
                cands_ids,
                1,
                rank_ids
            )

            rank_list = rank_list[
                :,
                :self.topk
            ]

            rec_ids.append(rank_list)

        if len(rec_ids) == 0:
            return np.empty(
                (0, self.topk),
                dtype=np.int64
            )

        rec_ids = torch.cat(
            rec_ids,
            dim=0
        )

        return rec_ids.cpu().numpy()

    def full_rank(self, u):
        """
        Rank all items for one user.
        """

        u = torch.tensor(
            [u],
            device=self.device,
            dtype=torch.long
        )

        user_emb = self.get_user_embedding(
            u
        ).squeeze(0)

        items_emb = self.embed_item.weight

        scores = torch.matmul(
            user_emb,
            items_emb.transpose(1, 0)
        )

        return torch.argsort(
            scores,
            descending=True
        )[:self.topk].cpu().numpy()
