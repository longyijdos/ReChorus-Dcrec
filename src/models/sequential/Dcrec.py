# -*- coding: UTF-8 -*-
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List
from utils import utils
from helpers.BaseReader import BaseReader
from models.model_utils import TransformerLayer, TransformerEmbedding
import math
import random
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from copy import deepcopy
from models.BaseModel import SequentialModel
import pickle
import zipfile
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from collections import defaultdict,Counter

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def cal_kl_1(target, input):
    target[target < 1e-8] = 1e-8
    target = torch.log(target + 1e-8)
    input = torch.log_softmax(input + 1e-8, dim=0)
    return F.kl_div(input, target, reduction='batchmean', log_target=True)


class CLLayer(torch.nn.Module):
    def __init__(self, num_hidden: int, tau: float = 0.5):
        super().__init__()
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_hidden)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        def f(x): return torch.exp(x / self.tau)

        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1

        def f(x): return torch.exp(x / self.tau)

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def vanilla_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        def f(x): return torch.exp(x / self.tau)

        pos_pairs = f(self.sim(z1, z2)).diag()
        neg_pairs = f(self.sim(z1, z2)).sum(1)
        return -torch.log(1e-8 + pos_pairs / neg_pairs)

    def vanilla_loss_with_one_negative(self, z1: torch.Tensor, z2: torch.Tensor):
        def f(x): return torch.exp(x / self.tau)

        pos_pairs = f(self.sim(z1, z2)).diag()
        neg_pairs = f(self.sim(z1, z2))
        rand_pairs = torch.randperm(neg_pairs.size(1))
        neg_pairs = neg_pairs[torch.arange(
            0, neg_pairs.size(0)), rand_pairs] + neg_pairs.diag()
        return -torch.log(pos_pairs / neg_pairs)

    def grace_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                   mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        return ret


def graph_dual_neighbor_readout(g: dgl.DGLGraph, aug_g: dgl.DGLGraph, node_ids, features):
    node_ids = node_ids.to(g.device)
    _, all_neighbors = g.out_edges(node_ids)
    all_nbr_num = g.out_degrees(node_ids)
    _, foreign_neighbors = aug_g.out_edges(node_ids)
    for_nbr_num = aug_g.out_degrees(node_ids)
    all_neighbors = [set(t.tolist())
                     for t in all_neighbors.split(all_nbr_num.tolist())]
    foreign_neighbors = [set(t.tolist())
                         for t in foreign_neighbors.split(for_nbr_num.tolist())]
    # sample foreign neighbors
    for i, nbrs in enumerate(foreign_neighbors):
        if len(nbrs) > 10:
            nbrs = random.sample(nbrs, 10)
            foreign_neighbors[i] = set(nbrs)
    civil_neighbors = [all_neighbors[i] - foreign_neighbors[i]
                       for i in range(len(all_neighbors))]
    # sample civil neighbors
    for i, nbrs in enumerate(civil_neighbors):
        if len(nbrs) > 10:
            nbrs = random.sample(nbrs, 10)
            civil_neighbors[i] = set(nbrs)
    for_lens = [len(t) for t in foreign_neighbors]
    cv_lens = torch.tensor([len(t)
                            for t in civil_neighbors], dtype=torch.int16)
    zero_indicies = (cv_lens == 0).nonzero().view(-1).tolist()
    cv_lens = cv_lens[cv_lens > 0].tolist()
    foreign_neighbors = torch.cat(
        [torch.tensor(list(s), dtype=torch.long) for s in foreign_neighbors])
    civil_neighbors = torch.cat(
        [torch.tensor(list(s), dtype=torch.long) for s in civil_neighbors])
    cv_feats = features[civil_neighbors].split(cv_lens)
    cv_feats = [t.mean(dim=0) for t in cv_feats]
    # insert zero vector for zero-length neighbors
    if len(zero_indicies) > 0:
        for i in zero_indicies:
            cv_feats.insert(i, torch.zeros_like(features[0]))
    for_feats = features[foreign_neighbors].split(for_lens)
    for_feats = [t.mean(dim=0) for t in for_feats]
    return torch.stack(cv_feats, dim=0), torch.stack(for_feats, dim=0)


def graph_augment(g: dgl.DGLGraph, user_ids, user_edges):
    # Augment the graph with the item sequence, deleting co-occurrence edges in the batched sequences
    # generating indicies like: [1,2] [2,3] ... as the co-occurrence rel.
    # indexing edge data using node indicies and delete them
    # for edge weights, delete them from the raw data using indexed edges
    user_ids = user_ids.cpu().numpy()
    node_indicies_a = np.concatenate(
        user_edges.loc[user_ids, "item_edges_a"].to_numpy())
    node_indicies_b = np.concatenate(
        user_edges.loc[user_ids, "item_edges_b"].to_numpy())
    node_indicies_a = torch.from_numpy(
        node_indicies_a).to(g.device)
    node_indicies_b = torch.from_numpy(
        node_indicies_b).to(g.device)
    edge_ids = g.edge_ids(node_indicies_a, node_indicies_b)

    aug_g: dgl.DGLGraph = deepcopy(g)
    # The features for the removed edges will be removed accordingly.
    aug_g.remove_edges(edge_ids)
    return aug_g


def graph_dropout(g: dgl.DGLGraph, keep_prob):
    # Firstly mask selected edge values, returns the true values along with the masked graph.
    origin_edge_w = g.edata['w']

    drop_size = int((1 - keep_prob) * g.num_edges())
    random_index = torch.randint(
        0, g.num_edges(), (drop_size,), device=g.device)
    mask = torch.zeros(g.num_edges(), dtype=torch.uint8,
                       device=g.device).bool()
    mask[random_index] = True
    g.edata['w'].masked_fill_(mask, 0)

    return origin_edge_w, g


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob=0.7):
        super(GCN, self).__init__()
        self.dropout_prob = dropout_prob
        self.layer = GraphConv(in_dim, out_dim, weight=False,
                               bias=False, allow_zero_in_degree=False)

    def forward(self, graph, feature):
        graph = dgl.add_self_loop(graph)
        origin_w, graph = graph_dropout(graph, 1 - self.dropout_prob)
        embs = [feature]
        for i in range(2):
            feature = self.layer(graph, feature, edge_weight=graph.edata['w'])
            F.dropout(feature, p=0.2, training=self.training)
            embs.append(feature)
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        # recover edge weight
        graph.edata['w'] = origin_w
        return final_emb


class Dcrec(SequentialModel):
    reader, runner = 'SeqReader', 'DcrecRunner'  # choose helpers in specific model classes

    def create_user_history_lists(self,corpus,mode='train'):
        """
        根据 BaseReader 实例创建 user_history_lists 字典。

        Args:
            base_reader (BaseReader): 包含用户交互数据的 BaseReader 实例。

        Returns:
            dict: 包含用户历史物品序列的字典，形式为 {user_id: [item_1, item_2, ...]}。
        """
        # 确保数据已加载
        if not hasattr(corpus, 'data_df'):
            raise ValueError("BaseReader instance must have 'data_df' loaded.")

        # 获取训练集数据
        train_data = corpus.data_df[mode]

        # 初始化 user_history_lists
        user_history_lists = {}

        # 按用户分组，提取交互历史
        for user_id, group in train_data.groupby('user_id'):
            # 按时间排序物品交互
            group = group.sort_values(by='time')
            item_sequence = group['item_id'].tolist()
            user_history_lists[user_id] = item_sequence

        return user_history_lists

    def build_sim_graph(self, mode='train'):
        k = 4
        row = []
        col = []
        for uid, item_seq in self.user_history_lists[mode].items():
            seq_len = len(item_seq)
            col.extend(item_seq)
            row.extend([uid] * seq_len)
        row = np.array(row)
        col = np.array(col)
        # n_users, n_items
        cf_graph = sp.csr_matrix(([1] * len(row), (row, col)), shape=(
            self.user_num + 1, self.item_num + 1), dtype=np.float32)
        similarity = cosine_similarity(cf_graph.transpose())
        # filter topk connections
        sim_items_slices = []
        sim_weights_slices = []
        i = 0
        while i < similarity.shape[0]:
            similarity = similarity[i:, :]
            sim = similarity[:256, :]
            sim_items = np.argpartition(sim, -(k + 1), axis=1)[:, -(k + 1):]
            sim_weights = np.take_along_axis(sim, sim_items, axis=1)
            sim_items_slices.append(sim_items)
            sim_weights_slices.append(sim_weights)
            i = i + 256
        sim = similarity[256:, :]
        sim_items = np.argpartition(sim, -(k + 1), axis=1)[:, -(k + 1):]
        sim_weights = np.take_along_axis(sim, sim_items, axis=1)
        sim_items_slices.append(sim_items)
        sim_weights_slices.append(sim_weights)

        sim_items = np.concatenate(sim_items_slices, axis=0)
        sim_weights = np.concatenate(sim_weights_slices, axis=0)
        row = []
        col = []
        for i in range(len(sim_items)):
            row.extend([i] * len(sim_items[i]))
            col.extend(sim_items[i])
        values = sim_weights / sim_weights.sum(axis=1, keepdims=True)
        values = np.nan_to_num(values).flatten()
        adj_mat = sp.csr_matrix((values, (row, col)), shape=(
            self.item_num + 1, self.item_num + 1))
        g = dgl.from_scipy(adj_mat, 'w')
        g.edata['w'] = g.edata['w'].float()
        return g

    def build_adj_graph(self, mode='train'):
        item_adj_dict = defaultdict(list)
        item_edges_of_user = dict()
        for uid, item_seq in self.user_history_lists[mode].items():
            item_edges_a, item_edges_b = [], []
            seq_len = len(item_seq)
            for i in range(seq_len):
                if i > 0:
                    item_adj_dict[item_seq[i]].append(item_seq[i - 1])
                    item_adj_dict[item_seq[i - 1]].append(item_seq[i])
                    item_edges_a.append(item_seq[i])
                    item_edges_b.append(item_seq[i - 1])
                if i + 1 < seq_len:
                    item_adj_dict[item_seq[i]].append(item_seq[i + 1])
                    item_adj_dict[item_seq[i + 1]].append(item_seq[i])
                    item_edges_a.append(item_seq[i])
                    item_edges_b.append(item_seq[i + 1])
            item_edges_of_user[uid] = (np.asarray(
                item_edges_a), np.asarray(item_edges_b))
        if mode == 'train':
            item_edges_of_user = pd.DataFrame.from_dict(
                item_edges_of_user, orient='index', columns=['item_edges_a', 'item_edges_b'])
        cols = []
        rows = []
        values = []
        for item in item_adj_dict:
            adj = item_adj_dict[item]
            adj_count = Counter(adj)
            rows.extend([item] * len(adj_count))
            cols.extend(adj_count.keys())
            values.extend(adj_count.values())

        adj_mat = sp.csr_matrix((values, (rows, cols)), shape=(
            self.item_num + 1, self.item_num + 1))
        adj_mat = adj_mat.tolil()
        adj_mat.setdiag(np.ones((self.item_num + 1,)))
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        g = dgl.from_scipy(norm_adj, 'w')
        g.edata['w'] = g.edata['w'].float()
        if mode == 'train':
            return g, item_edges_of_user
        else:
            return g, None

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, args, corpus, ):
        super().__init__(args, corpus)
        self.device = args.device
        self.emb_size = 64
        self.max_len = 50
        self.n_layers = 2
        self.n_heads = 2
        self.inner_size = 4 * self.emb_size
        self.dropout_rate = 0.1
        self.batch_size = 512
        self.weight_mean = 0.4
        self.kl_weight = 1.0e-2
        self.cl_lambda = 1.0e-4
        self.emb_layer = TransformerEmbedding(
            self.item_num + 1, self.emb_size, self.max_len)
        self.transformer_layers = nn.ModuleList([TransformerLayer(
            self.emb_size, self.n_heads, self.inner_size, self.dropout_rate) for _ in range(self.n_layers)])
        self.loss_func = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(self.dropout_rate)

        self.layernorm = nn.LayerNorm(self.emb_size, eps=1e-12)
        self.contrastive_learning_layer = CLLayer(
            self.emb_size, tau=0.8)
        self.attn_weights = nn.Parameter(
            torch.Tensor(self.emb_size, self.emb_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.emb_size))
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)
        self.user_history_lists={}
        for phase in ['train', 'dev', 'test']:
            self.user_history_lists[phase] = self.create_user_history_lists(corpus,phase)
        self.item_adjgraph={}
        self.item_simgraph={}
        self.user_edges = None
        for phase in ['train', 'dev', 'test']:
            if phase == 'train':
                self.item_adjgraph[phase], self.user_edges = self.build_adj_graph(phase)
            else:
                self.item_adjgraph[phase], _ = self.build_adj_graph(phase)
            self.item_simgraph[phase] = self.build_sim_graph(phase)
        self.graph_dropout = 0.3
        self.gcn = GCN(self.emb_size, self.emb_size, self.graph_dropout)
        self.loss_fct = nn.CrossEntropyLoss()
        # Initialize the model, optimizer, etc.
        self.apply(self._init_weights)

    """
	Key Methods
	"""

    def _subgraph_agreement(self, aug_g, adj_graph_emb, adj_graph_emb_last_items, last_items, feed_dict, mode):
        # here it firstly removes items of the sequence in the cooccurrence graph, and then performs the gnn aggregation, and finally calculates the item-wise agreement score.
        aug_output_seq = self.gcn_forward(g=aug_g)[last_items]
        civil_nbr_ro, foreign_nbr_ro = graph_dual_neighbor_readout(
            self.item_adjgraph[mode], aug_g, last_items, adj_graph_emb)

        view1_sim = F.cosine_similarity(
            adj_graph_emb_last_items, aug_output_seq, eps=1e-12)
        view2_sim = F.cosine_similarity(
            adj_graph_emb_last_items, foreign_nbr_ro, eps=1e-12)
        view3_sim = F.cosine_similarity(
            civil_nbr_ro, foreign_nbr_ro, eps=1e-12)
        agreement = (view1_sim + view2_sim + view3_sim) / 3
        agreement = torch.sigmoid(agreement)
        agreement = (agreement - agreement.min()) / \
                    (agreement.max() - agreement.min())
        agreement = (self.weight_mean / agreement.mean()) * agreement
        return agreement

    def get_attention_mask(self, item_seq, task_label=False):
        """Generate bidirectional attention mask for multi-head attention."""
        if task_label:
            label_pos = torch.ones((item_seq.size(0), 1), device=self.device)
            item_seq = torch.cat((label_pos, item_seq), dim=1)
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(
            1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def gcn_forward(self, g=None):
        item_emb = self.emb_layer.token_emb.weight
        item_emb = self.dropout(item_emb)
        g = g.to(item_emb.device)  # 将图移动到与特征相同的设备
        light_out = self.gcn(g, item_emb)
        return self.layernorm(light_out + item_emb)

    def forward_loss(self, feed_dict):
        # Construct batch_data
        batch_seqs = feed_dict
        max_seq_len = 50
        current_seq_len = batch_seqs.size(1)
        if current_seq_len < max_seq_len:
            padding_len = max_seq_len - current_seq_len
            batch_seqs = F.pad(batch_seqs, (0, padding_len), value=0)
        mask = (batch_seqs > 0).unsqueeze(1).repeat(
            1, batch_seqs.size(1), 1).unsqueeze(1)
        x = self.emb_layer(batch_seqs)
        for transformer in self.transformer_layers:
            x = transformer(x, mask)
        return x[:, -1, :]  # [B H]

    def forward(self, feed_dict, mode='test'):
        # Construct batch_data
        batch_user = feed_dict['user_id']
        batch_pos_items = feed_dict['item_id'][:, 0]  # Get the first item as the target item (真实项目)
        batch_items = feed_dict['item_id']  # All items (gt + sampled negative samples)
        batch_seqs = feed_dict['history_items']

        # Generate the sequence output
        seq_output = self.forward_loss(batch_seqs)
        last_items = batch_seqs[:, -1].view(-1)  # Get the last items in the sequences

        # Graph view
        adj_graph = self.item_adjgraph[mode]
        sim_graph = self.item_simgraph[mode]
        iadj_graph_output_raw = self.gcn_forward(adj_graph)
        iadj_graph_output_seq = iadj_graph_output_raw[last_items]
        isim_graph_output_seq = self.gcn_forward(sim_graph)[last_items]

        # Combine the different sources of information
        mixed_x = torch.stack(
            (seq_output, iadj_graph_output_seq, isim_graph_output_seq), dim=0)
        weights = (torch.matmul(mixed_x, self.attn_weights.unsqueeze(0)) * self.attn).sum(-1)
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        seq_output = (mixed_x * score).sum(0)

        # Get the embeddings for the specific items in feed_dict['item_id']
        item_indices = batch_items.view(-1)  # Flatten to 1D array of indices
        test_item_emb = self.emb_layer.token_emb(item_indices)  # [batch_size * num_items, H]

        # Reshape test_item_emb to [batch_size, num_items, H]
        batch_size, num_items = batch_items.size()
        test_item_emb = test_item_emb.view(batch_size, num_items, -1)

        # Ensure seq_output is [batch_size, 1, H] for bmm
        seq_output = seq_output.unsqueeze(1)

        # Calculate scores only for the selected items
        scores = torch.bmm(seq_output, test_item_emb.transpose(1, 2)).squeeze(1)  # [batch_size, num_items]



        return {'prediction': scores}

    def loss(self, feed_dict, mode='train'):
        batch_user = feed_dict['user_id']
        batch_pos_items = feed_dict['item_id'][:, 0]
        batch_seqs = feed_dict['history_items']
        last_items = batch_seqs[:, -1].view(-1)
        # graph view
        masked_g = self.item_adjgraph[mode]
        aug_g = graph_augment(self.item_adjgraph[mode], batch_user, self.user_edges)
        adj_graph_emb = self.gcn_forward(masked_g)
        sim_graph_emb = self.gcn_forward(self.item_simgraph[mode])
        adj_graph_emb_last_items = adj_graph_emb[last_items]
        sim_graph_emb_last_items = sim_graph_emb[last_items]

        seq_output = self.forward_loss(batch_seqs)
        aug_seq_output = self.forward_loss(batch_seqs)
        # First-stage CL, providing CL weights
        # CL weights from augmentation
        mainstream_weights = self._subgraph_agreement(
            aug_g, adj_graph_emb, adj_graph_emb_last_items, last_items, feed_dict, mode)
        # filtering those len=1, set weight=0.5
        seq_lens = batch_seqs.ne(0).sum(dim=1)
        mainstream_weights[seq_lens == 1] = 0.5

        expected_weights_distribution = torch.normal(self.weight_mean, 0.1, size=mainstream_weights.size()).to(
            self.device)
        kl_loss = self.kl_weight * cal_kl_1(expected_weights_distribution.sort()[0], mainstream_weights.sort()[0])

        personlization_weights = mainstream_weights.max() - mainstream_weights

        # contrastive learning
        cl_loss_adj = self.contrastive_learning_layer.vanilla_loss(
            aug_seq_output, adj_graph_emb_last_items)
        cl_loss_a2s = self.contrastive_learning_layer.vanilla_loss(
            adj_graph_emb_last_items, sim_graph_emb_last_items)
        cl_loss = (self.cl_lambda * (mainstream_weights *
                                     cl_loss_adj + personlization_weights * cl_loss_a2s)).mean()
        # Fusion After CL
        # 3, N_mask, dim
        mixed_x = torch.stack(
            (seq_output, adj_graph_emb[last_items], sim_graph_emb[last_items]), dim=0)
        weights = (torch.matmul(
            mixed_x, self.attn_weights.unsqueeze(0)) * self.attn).sum(-1)
        # 3, N_mask, 1
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        seq_output = (mixed_x * score).sum(0)
        # [item_num, H]
        test_item_emb = self.emb_layer.token_emb.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        batch_pos_items = batch_pos_items.to(torch.long)
        loss = self.loss_fct(logits + 1e-8, batch_pos_items)

        loss_dict = {
            "loss": loss.item(),
            "cl_loss": cl_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        return loss + cl_loss + kl_loss, loss_dict

    """
	Define Dataset Class
	"""

    class Dataset(SequentialModel.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)




