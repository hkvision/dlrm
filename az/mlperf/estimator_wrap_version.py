# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import builtins
import time
import json
# data generation

# numpy
import numpy as np

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import onnx

# pytorch
import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter

# For distributed run
import extend_distributed as ext_dist

import intel_pytorch_extension as ipex
from intel_pytorch_extension import core

# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag
# mixed-dimension trick
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver

import sklearn.metrics
import mlperf_logger

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

from torch.optim.lr_scheduler import _LRScheduler

exc = getattr(builtins, "IOError", "FileNotFoundError")


class LRPolicyScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
        self.num_warmup_steps = num_warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_start_step + num_decay_steps
        self.num_decay_steps = num_decay_steps

        if self.decay_start_step < self.num_warmup_steps:
            sys.exit("Learning rate warmup must finish before the decay starts")

        super(LRPolicyScheduler, self).__init__(optimizer)

    def get_lr(self):
        step_count = self._step_count
        if step_count < self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
            # decay
            decayed_steps = step_count - self.decay_start_step
            scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
            min_lr = 0.0000001
            lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            if self.num_decay_steps > 0:
                # freeze at last, either because we're after decay
                # or because we're between warmup and decay
                lr = self.last_lr
            else:
                # do not adjust
                lr = self.base_lrs
        return lr


class Cast(nn.Module):
    __constants__ = ['to_dtype']

    def __init__(self, to_dtype):
        super(Cast, self).__init__()
        self.to_dtype = to_dtype

    def forward(self, input):
        if input.is_mkldnn:
            return input.to_dense(self.to_dtype)
        else:
            return input.to(self.to_dtype)

    def extra_repr(self):
        return 'to(%s)' % self.to_dtype


### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            if self.use_ipex and self.bf16:
                LL = ipex.IpexMLPLinear(int(n), int(m), bias=True, output_stays_blocked=(i < ln.size - 2),
                                        default_blocking=32)
            else:
                LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)

            if self.bf16 and ipex.is_available():
                LL.to(torch.bfloat16)
            # prepack weight for IPEX Linear
            if hasattr(LL, 'reset_weight_shape'):
                LL.reset_weight_shape(block_for_dtype=torch.bfloat16)

            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                if self.bf16:
                    layers.append(Cast(torch.float32))
                layers.append(nn.Sigmoid())
            else:
                if self.use_ipex and self.bf16:
                    LL.set_activation_type('relu')
                else:
                    layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, local_ln_emb_sparse=None, ln_emb_dense=None):
        emb_l = nn.ModuleList()
        # save the numpy random state
        np_rand_state = np.random.get_state()
        emb_dense = nn.ModuleList()
        emb_sparse = nn.ModuleList()
        embs = range(len(ln))
        if local_ln_emb_sparse or ln_emb_dense:
            embs = local_ln_emb_sparse + ln_emb_dense
        for i in embs:
            # Use per table random seed for Embedding initialization
            np.random.seed(self.l_emb_seeds[i])
            n = ln[i]
            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(n, m, self.qr_collisions,
                                    operation=self.qr_operation, mode="sum", sparse=True)
            elif self.md_flag:
                base = max(m)
                _m = m[i] if n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)

            else:
                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                # approach 1
                if n >= self.sparse_dense_boundary:
                    EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True, _weight=torch.tensor(W, requires_grad=True))
                else:
                    EE = nn.EmbeddingBag(n, m, mode="sum", sparse=False, _weight=torch.tensor(W, requires_grad=True))
                # approach 2
                # EE.weight.data.copy_(torch.tensor(W))
                # approach 3
                # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
                if self.bf16 and ipex.is_available():
                    EE.to(torch.bfloat16)

            if ext_dist.my_size > 1:
                if n >= self.sparse_dense_boundary:
                    emb_sparse.append(EE)
                else:
                    emb_dense.append(EE)

            emb_l.append(EE)

        # Restore the numpy random state
        np.random.set_state(np_rand_state)
        return emb_l, emb_dense, emb_sparse

    def __init__(
            self,
            m_spa=None,
            ln_emb=None,
            ln_bot=None,
            ln_top=None,
            arch_interaction_op=None,
            arch_interaction_itself=False,
            sigmoid_bot=-1,
            sigmoid_top=-1,
            sync_dense_params=True,
            loss_threshold=0.0,
            ndevices=-1,
            qr_flag=False,
            qr_operation="mult",
            qr_collisions=0,
            qr_threshold=200,
            md_flag=False,
            md_threshold=200,
            bf16=False,
            use_ipex=False,
            sparse_dense_boundary=2048
    ):
        super(DLRM_Net, self).__init__()

        if (
                                (m_spa is not None)
                            and (ln_emb is not None)
                        and (ln_bot is not None)
                    and (ln_top is not None)
                and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.bf16 = bf16
            self.use_ipex = use_ipex
            self.sparse_dense_boundary = sparse_dense_boundary
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # generate np seeds for Emb table initialization
            self.l_emb_seeds = np.random.randint(low=0, high=100000, size=len(ln_emb))

            # If running distributed, get local slice of embedding tables
            if ext_dist.my_size > 1:
                n_emb = len(ln_emb)
                self.n_global_emb = n_emb
                self.rank = ext_dist.dist.get_rank()
                self.ln_emb_dense = [i for i in range(n_emb) if ln_emb[i] < self.sparse_dense_boundary]
                self.ln_emb_sparse = [i for i in range(n_emb) if ln_emb[i] >= self.sparse_dense_boundary]
                n_emb_sparse = len(self.ln_emb_sparse)
                self.n_local_emb_sparse, self.n_sparse_emb_per_rank = ext_dist.get_split_lengths(n_emb_sparse)
                self.local_ln_emb_sparse_slice = ext_dist.get_my_slice(n_emb_sparse)
                self.local_ln_emb_sparse = self.ln_emb_sparse[self.local_ln_emb_sparse_slice]
            # create operators
            if ndevices <= 1:
                if ext_dist.my_size > 1:
                    _, self.emb_dense, self.emb_sparse = self.create_emb(m_spa, ln_emb, self.local_ln_emb_sparse,
                                                                         self.ln_emb_dense)
                else:
                    self.emb_l, _, _ = self.create_emb(m_spa, ln_emb)

            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        need_padding = self.use_ipex and self.bf16 and x.size(0) % 2 == 1
        if need_padding:
            x = torch.nn.functional.pad(input=x, pad=(0, 0, 0, 1), mode='constant', value=0)
            ret = layers(x)
            return (ret[:-1, :])
        else:
            return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            E = emb_l[k]
            V = E(sparse_index_group_batch, sparse_offset_group_batch)

            ly.append(V)

        # print(ly)
        return ly

    def interact_features(self, x, ly):
        x = x.to(ly[0].dtype)
        if self.arch_interaction_op == "dot":
            if self.bf16:
                T = [x] + ly
                R = ipex.interaction(*T)
            else:
                # concatenate dense and sparse features
                (batch_size, d) = x.shape
                T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
                # perform a dot product
                Z = torch.bmm(T, torch.transpose(T, 1, 2))
                # append dense feature with the interactions (into a row vector)
                # approach 1: all
                # Zflat = Z.view((batch_size, -1))
                # approach 2: unique
                _, ni, nj = Z.shape
                # approach 1: tril_indices
                # offset = 0 if self.arch_interaction_itself else -1
                # li, lj = torch.tril_indices(ni, nj, offset=offset)
                # approach 2: custom
                offset = 1 if self.arch_interaction_itself else 0
                li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
                lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
                Zflat = Z[:, li, lj]
                # concatenate dense features and interactions
                R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, lS_i):
        if self.bf16:
            dense_x = dense_x.bfloat16()
        if ext_dist.my_size > 1:
            return self.distributed_forward(dense_x, lS_o, lS_i)
        elif self.ndevices <= 1:
            return self.sequential_forward(dense_x, lS_o, lS_i)
        else:
            return self.parallel_forward(dense_x, lS_o, lS_i)

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

    def distributed_forward(self, dense_x, lS_o, lS_i):
        batch_size = dense_x.size()[0]
        # WARNING: # of ranks must be <= batch size in distributed_forward call
        if batch_size < ext_dist.my_size:
            sys.exit("ERROR: batch_size (%d) must be larger than number of ranks (%d)" % (batch_size, ext_dist.my_size))

        lS_o_dense = [lS_o[i] for i in self.ln_emb_dense]
        lS_i_dense = [lS_i[i] for i in self.ln_emb_dense]
        lS_o_sparse = [lS_o[i] for i in self.ln_emb_sparse]  # partition sparse table in one group
        lS_i_sparse = [lS_i[i] for i in self.ln_emb_sparse]

        lS_i_sparse = ext_dist.shuffle_data(lS_i_sparse)
        g_i_sparse = [lS_i_sparse[:, i * batch_size:(i + 1) * batch_size].reshape(-1) for i in
                      range(len(self.local_ln_emb_sparse))]
        offset = torch.arange(batch_size * ext_dist.my_size).to(device)
        g_o_sparse = [offset for i in range(self.n_local_emb_sparse)]

        if (len(self.local_ln_emb_sparse) != len(g_o_sparse)) or (len(self.local_ln_emb_sparse) != len(g_i_sparse)):
            sys.exit("ERROR 0 : corrupted model input detected in distributed_forward call")
        # sparse embeddings
        ly_sparse = self.apply_emb(g_o_sparse, g_i_sparse, self.emb_sparse)
        a2a_req = ext_dist.alltoall(ly_sparse, self.n_sparse_emb_per_rank)
        # bottom mlp
        x = self.apply_mlp(dense_x, self.bot_l)
        # dense embeddings
        ly_dense = self.apply_emb(lS_o_dense, lS_i_dense, self.emb_dense)
        ly_sparse = a2a_req.wait()
        ly = ly_dense + list(ly_sparse)
        # interactions
        z = self.interact_features(x, ly)
        # top mlp
        p = self.apply_mlp(z, self.top_l)
        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(
                p, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z = p

        return z

    def parallel_forward(self, dense_x, lS_o, lS_i):
        ### prepare model (overwrite) ###
        # WARNING: # of devices must be >= batch size in parallel_forward call
        batch_size = dense_x.size()[0]
        ndevices = min(self.ndevices, batch_size, len(self.emb_l))
        device_ids = range(ndevices)
        # WARNING: must redistribute the model if mini-batch size changes(this is common
        # for last mini-batch, when # of elements in the dataset/batch size is not even
        if self.parallel_model_batch_size != batch_size:
            self.parallel_model_is_not_prepared = True

        if self.parallel_model_is_not_prepared or self.sync_dense_params:
            # replicate mlp (data parallelism)
            self.bot_l_replicas = replicate(self.bot_l, device_ids)
            self.top_l_replicas = replicate(self.top_l, device_ids)
            self.parallel_model_batch_size = batch_size

        if self.parallel_model_is_not_prepared:
            # distribute embeddings (model parallelism)
            t_list = []
            for k, emb in enumerate(self.emb_l):
                d = torch.device("cuda:" + str(k % ndevices))
                emb.to(d)
                t_list.append(emb.to(d))
            self.emb_l = nn.ModuleList(t_list)
            self.parallel_model_is_not_prepared = False

        ### prepare input (overwrite) ###
        # scatter dense features (data parallelism)
        # print(dense_x.device)
        dense_x = scatter(dense_x, device_ids, dim=0)
        # distribute sparse features (model parallelism)
        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit("ERROR: corrupted model input detected in parallel_forward call")

        t_list = []
        i_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            t_list.append(lS_o[k].to(d))
            i_list.append(lS_i[k].to(d))
        lS_o = t_list
        lS_i = i_list

        ### compute results in parallel ###
        # bottom mlp
        # WARNING: Note that the self.bot_l is a list of bottom mlp modules
        # that have been replicated across devices, while dense_x is a tuple of dense
        # inputs that has been scattered across devices on the first (batch) dimension.
        # The output is a list of tensors scattered across devices according to the
        # distribution of dense_x.
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
        # debug prints
        # print(x)

        # embeddings
        ly = self.apply_emb(lS_o, lS_i, self.emb_l)
        # debug prints
        # print(ly)

        # butterfly shuffle (implemented inefficiently for now)
        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on each device. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each device.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        if len(self.emb_l) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        t_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)
        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))
        # debug prints
        # print(ly)

        # interactions
        z = []
        for k in range(ndevices):
            zk = self.interact_features(x[k], ly[k])
            z.append(zk)
        # debug prints
        # print(z)

        # top mlp
        # WARNING: Note that the self.top_l is a list of top mlp modules that
        # have been replicated across devices, while z is a list of interaction results
        # that by construction are scattered across devices on the first (batch) dim.
        # The output is a list of tensors scattered across devices according to the
        # distribution of z.
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)

        ### gather the distributed results ###
        p0 = gather(p, self.output_d, dim=0)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0

        return z0


def data_creator(config, data_size):
    import random
    from torch.utils.data import Dataset, DataLoader
    records = []
    for i in range(10000):
        label = random.randint(0, 1)
        int_features = [random.randint(0, 100) for i in range(13)]
        cat_features = []
        for i in range(26):
            cat_features.append(random.randint(0, config["ln_emb"][i] -1))
        records.append([label, int_features, cat_features])
    records = records * data_size
    X_int = np.array([x[1] for x in records], dtype=np.int32)
    X_cat = np.array([x[2] for x in records], dtype=np.int32)
    y = np.array([x[0] for x in records], dtype=np.int32)
    x = {"X_int": X_int, "X_cat": X_cat}
    print("Data size on worker: ", len(y))

    class NDArrayDataset(Dataset):
        def __init__(self, x, y):
            self.x = x  # features
            self.y = y  # labels

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            x_i = {}
            for k, v in self.x.items():
                x_i[k] = v[i]
            y_i = self.y[i]
            return x_i, y_i

    dataset = NDArrayDataset(x, y)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=config["collate_fn"],
    )
    return loader


def train_data_creator(config):
    return data_creator(config, data_size=100)


def test_data_creator(config):
    return data_creator(config, data_size=10)


def model_creator(config):
    print('Creating the model...')
    dlrm = DLRM_Net(
        config["m_spa"],
        config["ln_emb"],
        config["ln_bot"],
        config["ln_top"],
        arch_interaction_op=config["arch_interaction_op"],
        arch_interaction_itself=config["arch_interaction_itself"],
        sigmoid_bot=-1,
        sigmoid_top=config["ln_top"].size - 2,
        sync_dense_params=config["sync_dense_params"],
        loss_threshold=config["loss_threshold"],
        ndevices=config["ndevices"],
        qr_flag=config["qr_flag"],
        qr_operation=config["qr_operation"],
        qr_collisions=config["qr_collisions"],
        qr_threshold=config["qr_threshold"],
        md_flag=config["md_flag"],
        md_threshold=config["md_threshold"],
        sparse_dense_boundary=config["sparse_dense_boundary"],
        bf16=config["bf16"],
        use_ipex=config["use_ipex"])
    print('Model created!')
    return dlrm


def optimizer_creator(model, config):
    optimizer = torch.optim.SGD([
        {"params": [p for emb in model.emb_sparse for p in emb.parameters()],
         "lr": args.learning_rate / ext_dist.my_size},
        {"params": [p for emb in model.emb_dense for p in emb.parameters()], "lr": args.learning_rate},
        {"params": model.bot_l.parameters(), "lr": args.learning_rate},
        {"params": model.top_l.parameters(), "lr": args.learning_rate}
    ], lr=config["learning_rate"])
    return optimizer


def scheduler_creator(optimizer, config):
    return LRPolicyScheduler(optimizer, config["lr_num_warmup_steps"], config["lr_decay_start_step"],
                             config["lr_num_decay_steps"])


if __name__ == "__main__":
    # the reference implementation doesn't clear the cache currently
    # but the submissions are required to do that
    mlperf_logger.log_event(key=mlperf_logger.constants.CACHE_CLEAR, value=True)

    mlperf_logger.log_start(key=mlperf_logger.constants.INIT_START, log_all_ranks=True)

    ### import packages ###
    import sys
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument("--arch-embedding-size", type=str, default="4-3-2")
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=str, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=str, default="4-2-1")
    parser.add_argument("--arch-interaction-op", type=str, default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument("--loss-weights", type=str, default="1.0-1.0")  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # distributed run
    parser.add_argument("--dist-backend", type=str, default="")
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
    parser.add_argument("--mlperf-bin-shuffle", action='store_true', default=False)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)
    # embedding table is sparse table only if sparse_dense_boundary >= 2048
    parser.add_argument("--sparse-dense-boundary", type=int, default=2048)
    # bf16 option
    parser.add_argument("--bf16", action='store_true', default=False)
    # ipex option
    parser.add_argument("--use-ipex", action="store_true", default=False)

    args = parser.parse_args()

    if args.mlperf_logging:
        print('command line args: ', json.dumps(vars(args)))

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if (args.test_mini_batch_size < 0):
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if (args.test_num_workers < 0):
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()
    use_ipex = args.use_ipex
    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda", 0)
        ngpus = torch.cuda.device_count()  # 1
        print("Using {} GPU(s)...".format(ngpus))
    elif use_ipex:
        device = torch.device("dpcpp")
        print("Using IPEX...")
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data

    mlperf_logger.barrier()
    mlperf_logger.log_end(key=mlperf_logger.constants.INIT_STOP)
    mlperf_logger.barrier()
    mlperf_logger.log_start(key=mlperf_logger.constants.RUN_START)
    mlperf_logger.barrier()

    with np.load("train_fea_count.npz") as data:
        ln_emb = data["counts"]

    m_den = 13  # number of dense features
    ln_bot[0] = 13

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    num_fea = ln_emb.size + 1  # num sparse + num dense features
    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # assign mixed dimensions if applicable
    if args.md_flag:
        m_spa = md_solver(
            torch.tensor(ln_emb),
            args.md_temperature,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims
        ).tolist()

    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    def collate_wrapper_criteo(list_of_tuples):
        # where each tuple is ({"X_int": ndarray, "X_cat": ndarray}, y)
        # Or can change to list [ndarray, ndarray] for X_int and X_cat in order.
        transposed_data = list(zip(*list_of_tuples))
        X_int_list = [x["X_int"] for x in transposed_data[0]]
        X_cat_list = [x["X_cat"] for x in transposed_data[0]]
        X_int = torch.log(torch.tensor(X_int_list, dtype=torch.float) + 1)
        X_cat = torch.tensor(X_cat_list, dtype=torch.long)
        T = torch.tensor(transposed_data[1], dtype=torch.float32).view(-1, 1)

        batchSize = X_cat.shape[0]
        featureCnt = X_cat.shape[1]

        lS_i = [X_cat[:, i] for i in range(featureCnt)]
        lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]

        return X_int, torch.stack(lS_o), torch.stack(lS_i), T

    config = vars(args)
    config.update({"m_spa": m_spa, "ln_emb": ln_emb, "ln_bot": ln_bot, "ln_top": ln_top})
    config["ndevices"] = ndevices
    config["collate_fn"] = collate_wrapper_criteo

    from mpi_estimator import MPIEstimator
    estimator = MPIEstimator(
        model_creator=model_creator,
        optimizer_creator=optimizer_creator,
        loss_creator=torch.nn.BCELoss(),
        scheduler_creator=scheduler_creator,
        config=config,
        workers_per_node=2,
        hosts=["172.168.3.106", "172.168.3.107"],
        env={"KMP_BLOCKTIME": "1",
             "KMP_AFFINITY": "granularity=fine,compact,1,0",
             "CCL_WORKER_COUNT": "4",
             "CCL_WORKER_AFFINITY": "0,1,2,3,24,25,26,27",
             "CCL_ATL_TRANSPORT": "ofi"})
    estimator.fit(train_data_creator, epochs=args.nepochs,
                  batch_size=args.mini_batch_size)
