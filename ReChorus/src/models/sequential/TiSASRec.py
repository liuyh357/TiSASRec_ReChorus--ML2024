# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" TiSASRec
Reference:
    "Time Interval Aware Self-Attention for Sequential Recommendation"
    Jiacheng Li et al., WSDM'2020.
CMD example:
    python main.py --model_name TiSASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 \
    --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn
import numpy as np
from models.BaseModel import SequentialModel


class TiSASRec(SequentialModel):
    """TiSASRec Model class, a time-aware self-attention model for sequential recommendation."""
    reader = 'SeqReader'  # Reader for the dataset
    runner = 'BaseRunner'  # Runner for the training process
    extra_log_args = ['emb_size', 'num_layers', 'num_heads', 'time_max']  # Additional logging arguments

    @staticmethod
    def parse_model_args(parser):
        """
        Add model-specific arguments to the parser.
        Args:
            parser: Argument parser
        Returns:
            parser with additional arguments
        """
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1, help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads.')
        parser.add_argument('--time_max', type=int, default=512, help='Max time intervals.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        """
        Initialize TiSASRec model.
        Args:
            args: Arguments for the model configuration
            corpus: The dataset corpus
        """
        super().__init__(args, corpus)
        self.emb_size = args.emb_size  # Embedding size
        self.max_his = args.history_max  # Maximum history length
        self.num_layers = args.num_layers  # Number of transformer layers
        self.num_heads = args.num_heads  # Number of attention heads
        self.max_time = args.time_max  # Maximum time for intervals
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)

        # Calculate the minimum interval for each user in the dataset
        self.user_min_interval = self._compute_user_min_intervals(corpus)

        # Define model parameters (embeddings and transformer layers)
        self._define_params()
        self.apply(self.init_weights)

    def _compute_user_min_intervals(self, corpus):
        """Compute the minimum time interval for each user in the dataset."""
        user_min_interval = {}
        for u, user_df in corpus.all_df.groupby('user_id'):
            time_seqs = user_df['time'].values
            interval_matrix = np.abs(time_seqs[:, None] - time_seqs[None, :])
            min_interval = np.min(interval_matrix + (interval_matrix <= 0) * 0xFFFFFFFF)  # Avoid zero intervals
            user_min_interval[u] = min_interval
        return user_min_interval

    def _define_params(self):
        """Define the model's parameter layers (embeddings, transformer blocks)."""
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)  # Item embeddings
        self.p_k_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)  # Position key embeddings
        self.p_v_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)  # Position value embeddings
        self.t_k_embeddings = nn.Embedding(self.max_time + 1, self.emb_size)  # Time key embeddings
        self.t_v_embeddings = nn.Embedding(self.max_time + 1, self.emb_size)  # Time value embeddings

        # Define transformer blocks
        self.transformer_block = nn.ModuleList([
            TimeIntervalTransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
                                         dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])

    def forward(self, feed_dict):
        """
        Forward pass of the TiSASRec model.
        Args:
            feed_dict: Dictionary containing input data
        Returns:
            A dictionary with 'prediction' as the model output
        """
        # Extract input data
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        i_history = feed_dict['history_items']  # [batch_size, history_max]
        t_history = feed_dict['history_times']  # [batch_size, history_max]
        user_min_t = feed_dict['user_min_intervals']  # [batch_size]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = i_history.shape

        valid_his = (i_history > 0).long()
        his_vectors = self.i_embeddings(i_history)  # Item embeddings for history

        # Position and interval embeddings
        pos_k, pos_v = self._get_position_embeddings(lengths, seq_len, valid_his)
        inter_k, inter_v = self._get_interval_embeddings(t_history, user_min_t, batch_size, seq_len)

        # Apply transformer blocks with self-attention
        attn_mask = torch.from_numpy(np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))).to(self.device)
        for block in self.transformer_block:
            his_vectors = block(his_vectors, pos_k, pos_v, inter_k, inter_v, attn_mask)

        his_vectors = his_vectors * valid_his[:, :, None].float()
        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]  # Select last valid vector

        # Calculate prediction based on item embeddings
        i_vectors = self.i_embeddings(i_ids)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        return {'prediction': prediction.view(batch_size, -1)}

    def _get_position_embeddings(self, lengths, seq_len, valid_his):
        """
        Compute position embeddings for the input sequence.
        Args:
            lengths: Sequence lengths for each batch
            seq_len: Maximum sequence length
            valid_his: A mask indicating valid history items
        Returns:
            pos_k, pos_v: Position key and value embeddings
        """
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_k = self.p_k_embeddings(position)
        pos_v = self.p_v_embeddings(position)
        return pos_k, pos_v

    def _get_interval_embeddings(self, t_history, user_min_t, batch_size, seq_len):
        """
        Compute interval embeddings based on time differences.
        Args:
            t_history: Time history of items
            user_min_t: Minimum time interval for each user
            batch_size: Batch size
            seq_len: Sequence length
        Returns:
            inter_k, inter_v: Interval key and value embeddings
        """
        interval_matrix = (t_history[:, :, None] - t_history[:, None, :]).abs()
        interval_matrix = (interval_matrix / user_min_t.view(-1, 1, 1)).long().clamp(0, self.max_time)
        inter_k = self.t_k_embeddings(interval_matrix)
        inter_v = self.t_v_embeddings(interval_matrix)
        return inter_k, inter_v

    # ----- 新增的未使用函数（示例） -----

    def _compute_average_item_frequency(self, corpus):
        """计算每个物品的平均出现频率"""
        item_counts = corpus.all_df.groupby('item_id').size()
        total_items = len(corpus.all_df)
        item_freq = item_counts / total_items
        return item_freq

    def _get_attention_weights(self, attention_scores, mask):
        """
        计算注意力权重。
        此函数尚未使用，但它可以在计算自注意力时提供更详细的注意力权重。
        Args:
            attention_scores: 注意力分数
            mask: Attention mask
        Returns:
            归一化后的注意力权重
        """
        attention_weights = attention_scores.masked_fill(mask == 0, -np.inf)  # 应用mask
        attention_weights = (attention_weights - attention_weights.max()).softmax(dim=-1)  # 归一化
        return attention_weights

    def _calculate_loss(self, prediction, target):
        """
        损失计算函数（未使用）。
        可以作为未来扩展的一部分来计算损失。
        Args:
            prediction: 预测值
            target: 真实值
        Returns:
            损失值
        """
        loss = nn.MSELoss()(prediction, target)  # 示例使用均方误差损失
        return loss

    def _apply_dropout_to_embeddings(self, embeddings, dropout_rate=0.1):
        """
        应用dropout到嵌入层（未使用）。
        这可以用于增加模型的泛化能力。
        Args:
            embeddings: 输入的嵌入向量
            dropout_rate: Dropout的比率
        Returns:
            应用dropout后的嵌入向量
        """
        dropout = nn.Dropout(dropout_rate)
        return dropout(embeddings)


class TimeIntervalMultiHeadAttention(nn.Module):
    """
    时间间隔感知的多头自注意力层
    """
    def __init__(self, d_model, n_heads, kq_same=False, bias=True):
        super().__init__()
        self.d_model = d_model
        self.h = n_heads
        self.d_k = self.d_model // self.h
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q, k, v, pos_k, pos_v, inter_k, inter_v, mask):
        """
        多头注意力机制
        Args:
            q, k, v: 输入的query, key, value张量
            pos_k, pos_v: 位置编码
            inter_k, inter_v: 时间间隔编码
            mask: 掩码矩阵
        """
        bs, seq_len = k.size(0), k.size(1)

        k = (self.k_linear(k) + pos_k).view(bs, seq_len, self.h, self.d_k)
        if not self.kq_same:
            q = self.q_linear(q).view(bs, seq_len, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, seq_len, self.h, self.d_k)
        v = (self.v_linear(v) + pos_v).view(bs, seq_len, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        inter_k = inter_k.view(bs, seq_len, seq_len, self.h, self.d_k)
        inter_v = inter_v.view(bs, seq_len, seq_len, self.h, self.d_k)
        inter_k = inter_k.transpose(2, 3).transpose(1, 2)
        inter_v = inter_v.transpose(2, 3).transpose(1, 2)

        output = self.scaled_dot_product_attention(q, k, v, inter_k, inter_v, self.d_k, mask)
        output = output.transpose(1, 2).reshape(bs, -1, self.d_model)
        return output

    @staticmethod
    def scaled_dot_product_attention(q, k, v, inter_k, inter_v, d_k, mask):
        """
        计算加权注意力分数
        """
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores += (q[:, :, :, None, :] * inter_k).sum(-1)
        scores = scores / d_k ** 0.5
        scores.masked_fill_(mask == 0, -np.inf)
        scores = (scores - scores.max()).softmax(dim=-1)
        output = torch.matmul(scores, v)
        output += (scores[:, :, :, :, None] * inter_v).sum(-2)
        return output


class TimeIntervalTransformerLayer(nn.Module):
    """
    包含时间间隔感知自注意力的Transformer层
    """
    def __init__(self, d_model, d_ff, n_heads, dropout, kq_same=False):
        super().__init__()
        self.masked_attn_head = TimeIntervalMultiHeadAttention(d_model, n_heads, kq_same=kq_same)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, seq, pos_k, pos_v, inter_k, inter_v, mask):
        context = self.masked_attn_head(seq, seq, seq, pos_k, pos_v, inter_k, inter_v, mask)
        context = self.layer_norm1(self.dropout1(context) + seq)
        output = self.linear1(context).relu()
        output = self.linear2(output)
        output = self.layer_norm2(self.dropout2(output) + context)
        return output
