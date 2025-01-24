import torch
import torch.nn as nn
from typing import List



class CFModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.cf_emb_size = args.cf_emb_size
        self.max_his_len = args.max_his_len
        self.args = args

        self.item_cf_emb = self.get_cf_emb()
        self.item_emb_layer = nn.Embedding.from_pretrained(self.item_cf_emb)

        self.position_emb = PositionalEmbedding(self.max_his_len,
                                           self.cf_emb_size)

        self.transformer = Transformer(emb_size=self.cf_emb_size,
                                           num_heads=self.num_heads,
                                           num_layers=self.num_layers,
                                           dropout=self.dropout)


        self.attn_pooling = Target_Attention(self.cf_emb_size,
                                                     self.cf_emb_size)

        self._init_weights()
        self.to(self.args.device)

    def _init_weights(self):
        # weight initialization xavier_normal (a.k.a glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Embedding):
                continue

    def get_cf_emb(self):
        # load item emb .tar file
        cf_emb = torch.load(self.args.item_emb_path)
        item_cf_emb = cf_emb['embedding_item.weight'].cpu().detach()
        item_cf_emb = torch.cat([torch.zeros(1, 64), item_cf_emb], 0)
        return item_cf_emb
    
    def forward(self, all_his, target_item):
        # get cf emb of all_his
        all_his_emb = self.item_emb_layer(all_his)
        target_emb = self.item_emb_layer(target_item).squeeze(1)
        all_his_mask = (all_his == 0).bool()


        # get positional emb of all_his
        # all_his_emb_w_pos = all_his_emb
        behavior_emb = self.transformer(all_his_emb, all_his_mask)

        # get attention pooling of all_his
        all_his_attn = self.attn_pooling(behavior_emb, target_emb, all_his_mask)

        return all_his_attn



class Transformer(nn.Module):
    def __init__(self, emb_size, num_heads, num_layers, dropout) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=emb_size,
            dropout=dropout,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformerEncoderLayer, num_layers=num_layers)

    def forward(self,
                his_emb: torch.Tensor,
                src_key_padding_mask: torch.Tensor,
                src_mask: torch.Tensor = None):
        if src_mask is not None:
            src_mask_expand = src_mask.unsqueeze(1).expand(
                (-1, self.num_heads, -1, -1)).reshape(
                    (-1, his_emb.size(1), his_emb.size(1)))
            his_encoded = self.transformer_encoder(
                src=his_emb,
                src_key_padding_mask=src_key_padding_mask,
                mask=src_mask_expand)
        else:
            his_encoded = self.transformer_encoder(
                src=his_emb, src_key_padding_mask=src_key_padding_mask)

        return his_encoded

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim)
        nn.init.xavier_normal_(self.pe.weight.data)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
    

class Target_Attention(nn.Module):
    def __init__(self, hid_dim1, hid_dim2):
        super().__init__()

        self.W = nn.Parameter(torch.randn((1, hid_dim1, hid_dim2)))
        nn.init.xavier_normal_(self.W)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq_emb, target, mask):
        score = torch.matmul(seq_emb, self.W)
        score = torch.matmul(score, target.unsqueeze(-1))

        all_score = score.masked_fill(mask.unsqueeze(-1), torch.tensor(-1e16))
        all_weight = self.softmax(all_score.transpose(-2, -1))
        all_vec = torch.matmul(all_weight, seq_emb).squeeze(1)

        return all_vec

class Target_Attention(nn.Module):
    def __init__(self, hid_dim1, hid_dim2):
        super().__init__()

        self.W = nn.Parameter(torch.randn((1, hid_dim1, hid_dim2)))
        nn.init.xavier_normal_(self.W)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq_emb, target, mask):
        score = torch.matmul(seq_emb, self.W)
        score = torch.matmul(score, target.unsqueeze(-1))

        all_score = score.masked_fill(mask.unsqueeze(-1), torch.tensor(-1e16))
        all_weight = self.softmax(all_score.transpose(-2, -1))
        all_vec = torch.matmul(all_weight, seq_emb).squeeze(1)

        return all_vec

class Target_Attention_Sem(nn.Module):
    def __init__(self, hid_dim1, hid_dim2):
        super().__init__()

        self.W = nn.Parameter(torch.randn((1, hid_dim1, hid_dim2)))
        nn.init.xavier_normal_(self.W)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq_emb, target, mask, target_attn):
        score = torch.matmul(seq_emb, self.W)
        # target = target.mean(dim=1) 
        target_mask = (target_attn == 1).float()
        masked_target = target * target_mask.unsqueeze(-1)
        target = masked_target.sum(dim=1) / (target_mask.sum(dim=1, keepdim=True) + 1e-10)
        
        score = torch.matmul(score, target.unsqueeze(-1))

        all_score = score.masked_fill(mask.unsqueeze(-1), torch.tensor(-1e16))
        all_weight = self.softmax(all_score.transpose(-2, -1))
        all_vec = torch.matmul(all_weight, seq_emb)

        return all_vec, target.unsqueeze(1)