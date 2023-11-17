import torch
import torch.nn as nn
import torch.nn.functional as F
import IPython
from model import STGCN
from model import utils_rf

# from utils_rf import scaled_dot_product, expand_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=634):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float().cuda()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        ).cuda()
        self.positional_encoding = torch.zeros((1, max_len, d_model)).cuda()
        self.positional_encoding[0, :, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[0, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return x + self.positional_encoding[:, : x.size(1)].detach()


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        print(x.size())
        batch_size, _, seq_length, _ = x.size()
        if mask is not None:
            mask = utils_rf.expand_mask(mask)
        qkv = self.qkv_proj(x)
        print("the shape of qkv", qkv.shape)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = utils_rf.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.fc_query = nn.Linear(d_model, d_model)
        self.fc_key = nn.Linear(d_model, d_model)
        self.fc_value = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None, desired_portion=0.9):
        # Linear transformations
        query = self.fc_query(query)
        key = self.fc_key(key)
        value = self.fc_value(value)

        # Reshape for multi-heads
        query = query.view(query.shape[0], -1, self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        key = key.view(key.shape[0], -1, self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        value = value.view(value.shape[0], -1, self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )

        # Attention calculation
        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / (self.head_dim**0.5)
        # print("energy's sample1, head1 is", energy[0][0])
        # IPython.embed()
        # sys.exit()
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        print("energy shape is ", energy.shape)

        desired_frames = round(desired_portion * energy.shape[-1])
        energy_clusters = utils_rf.eliminate_frames_4d(energy, desired_frames)

        attention = torch.softmax(energy, dim=-1)
        print("attention's shape is", attention.shape)

        x = torch.matmul(attention, value)

        # Reshape and concatenate
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], -1, self.n_heads * self.head_dim)

        # Final linear layer
        x = self.fc_out(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attention_output = self.attention(x, x, x, mask)
        print(
            "shape of attention_output and x_input for transformer block{}, {}".format(
                attention_output.shape, x.shape
            )
        )
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        feed_forward_output = self.feed_forward(x)
        x = x + self.dropout(feed_forward_output)
        x = self.norm2(x)
        print("shape of xoutput in the end of transformer block ", x.shape)

        return x, attention_output


class AttentionBranch(nn.Module):
    def __init__(
        self,
        config,
        num_classes,
        num_att_edge,
        s_kernel_size,
        t_kernel_size,
        dropout,
        A_size,
    ):
        super(AttentionBranch, self).__init__()
        # STGC-Block config
        kwargs = dict(
            s_kernel_size=s_kernel_size,
            t_kernel_size=t_kernel_size,
            dropout=dropout,
            A_size=A_size,
        )
        self.stgc_block1 = STGCN.STGC_Block(
            config[0][0], config[0][1], config[0][2], **kwargs
        )
        self.stgc_block2 = STGCN.STGC_Block(
            config[1][0], config[1][1], config[1][2], **kwargs
        )
        self.stgc_block3 = STGCN.STGC_Block(
            config[2][0], config[2][1], config[2][2], **kwargs
        )

        # Prediction
        self.fc = nn.Conv2d(config[-1][1], num_classes, kernel_size=1, padding=0)

        # Attention
        self.att_bn = nn.BatchNorm2d(config[-1][1])
        self.att_conv = nn.Conv2d(
            config[-1][1], num_classes, kernel_size=1, padding=0, stride=1, bias=False
        )

        # Attention node
        self.att_node_conv = nn.Conv2d(
            num_classes, 1, kernel_size=1, padding=0, stride=1, bias=False
        )
        self.att_node_bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

        # Attention edge
        self.num_att_edge = num_att_edge
        self.att_edge_conv = nn.Conv2d(
            num_classes,
            num_att_edge * A_size[2],
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )
        self.att_edge_bn = nn.BatchNorm2d(num_att_edge * A_size[2])
        self.transformer_bn = nn.BatchNorm1d(num_features=634)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        # Attention frame pairs
        # frame* frame
        # self.num_att_frame = 2533
        # self.att_frame_conv = nn.Conv2d(
        #     num_classes,
        #     self.num_att_frame * self.num_att_frame,
        #     kernel_size=1,
        #     padding=0,
        #     stride=1,
        #     bias=False,
        # )
        # self.att_frame_bn = nn.BatchNorm2d(self.num_att_frame**2)

        # Transformer
        self.transformer1 = TransformerBlock(d_model=1664, n_heads=4, d_ff=128)
        self.positional_encoding = PositionalEncoding(1000)

        # Multi-head self-attention encoder block x1
        # self.multi_head = EncoderBlock(64, 8, 64, dropout=0.2)

    def forward(self, x, A):
        print("shape of X after the feature extraction: ", x.shape)
        N, c, T, V = x.size()

        # STGC Block
        x = self.stgc_block1(x, A, None)
        x = self.stgc_block2(x, A, None)
        x = self.stgc_block3(x, A, None)

        # Prediction
        x_out = F.avg_pool2d(x, x.size()[2:])
        x_out = x_out.view(N, -1, 1, 1)
        x_out = self.fc(x_out)
        output = x_out.view(x_out.size(0), -1)

        # Attention
        x_att = self.att_bn(x)
        x = x.permute(0, 2, 1, 3).contiguous().flatten(start_dim=2, end_dim=3)
        print("the x size is: after 3 stgc in Att branch", x.shape)
        x = self.transformer_bn(x)
        # IPython.embed()

        # self attention
        print("shape of x_att: ", x_att.shape)
        _, att_mat = self.transformer1(x)
        print("shape of att_mat: ", att_mat.shape)
        # x_att = x_att.permute(0, 3, 2, 1).contiguous()

        # attn_mtx = self.multi_head(x_att)
        # print("shape of attn_mtx: ", attn_mtx.shape)
        x_att = self.att_conv(x_att)

        # Attention node
        x_node = self.att_node_conv(x_att)
        x_node = self.att_node_bn(x_node)
        x_node = F.interpolate(x_node, size=(T, V))
        att_node = self.sigmoid(x_node)
        print("attnodes shape are: ", att_node.shape)
        # print("attnodes are: ", att_node)

        # Attention edge
        print("x_att shape are: ", x_att.shape)
        x_edge = F.avg_pool2d(x_att, (x_att.size()[2], 1))
        print("xedges before conv shape are: ", x_edge.shape)
        x_edge = self.att_edge_conv(x_edge)

        x_edge = self.att_edge_bn(x_edge)
        print("xedges shape are: ", x_edge.shape)
        x_edge = x_edge.view(N, self.num_att_edge, V, V)
        x_edge = self.tanh(x_edge)
        att_edge = self.relu(x_edge)
        print("attedges shape are: ", att_edge.shape)
        # print("attedges are: ", att_edge)

        # Attention Frame
        # x_fpair = self.att_node_conv(x_att)
        # x_fpair = self.att_node_bn(x_fpair)
        # x_fpair = F.interpolate(x_fpair, size=(T, T))
        # att_frame = self.sigmoid(x_fpair)
        # print("attframes shape are: ", att_frame.shape)
        # print("attframes are: ", att_frame)

        return output, att_node, att_edge  # , attframe


class FeatureExtractor(nn.Module):
    def __init__(self, config, s_kernel_size, t_kernel_size, dropout, A_size):
        super(FeatureExtractor, self).__init__()
        # Batch Normalization
        self.bn = nn.BatchNorm1d(config[0][0] * A_size[2])

        # STGC-Block config
        kwargs = dict(
            s_kernel_size=s_kernel_size,
            t_kernel_size=t_kernel_size,
            dropout=dropout,
            A_size=A_size,
        )
        self.stgc_block1 = STGCN.STGC_Block(
            config[0][0], config[0][1], config[0][2], **kwargs
        )
        self.stgc_block2 = STGCN.STGC_Block(
            config[1][0], config[1][1], config[1][2], **kwargs
        )
        self.stgc_block3 = STGCN.STGC_Block(
            config[2][0], config[2][1], config[2][2], **kwargs
        )

    def forward(self, x, A):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        print("after batch norm", x.shape)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        # STGC Blocks
        x = self.stgc_block1(x, A, None)
        x = self.stgc_block2(x, A, None)
        x = self.stgc_block3(x, A, None)

        print("after feature extractor", x.shape)
        return x
