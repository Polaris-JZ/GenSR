'''
    Semantic Filtering Model
    Purpose: Filter out the item that is not related to the query and target item
    Input:
        query: str, query sentence [bs, 1]
        target item: str, target item [bs, 1]
        item history: list of str, history items [bs, item_len]
    Output:
        filtered item history: list of str, filtered history items [bs, filter_item_num]
    Hyper-parameters:
        filter_threshold: float, threshold for the importance score
        filter_item_num: int, number of items leave after filtering if filtered item num is bigger than filter_item_num
    Model Structure:
        - BERT-based encoder for the query and target item (concat the query and target item for input)
        - BERT-based encoder for the item history
        - mean pooling for the query and target item
        - calculate the attention score between query/target item and item history -> get vector representation for the history items
        - Linear layer to get the importance score for each item
        - Filter out the item according to the importance score and filter_item_num
        - Output the filtered item history
'''
import logging
import torch
from torch import nn
from model.CFModel import Transformer
from model.CFModel import Target_Attention_Sem
import time
class SemModel(nn.Module):
    def __init__(self, args, model, tokenizer):
        super(SemModel, self).__init__()
        self.filter_threshold = args.filter_threshold
        self.filter_item_num = args.filter_item_num
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.transformer = Transformer(emb_size=self.model.config.hidden_size,
                                           num_heads=self.args.num_heads,
                                           num_layers=self.args.num_layers,
                                           dropout=self.args.dropout)

        self.attn_pooling = Target_Attention_Sem(self.model.config.hidden_size,
                                                     self.model.config.hidden_size)
            
    def forward(self, target_item, target_item_attn, item_history_id, item_history_attn, item_num):
        item_num = item_num.to(self.model.device)
        batch_size = len(target_item)

        history_emb = self.model.shared(item_history_id.to(self.model.device))
        history_mask = (item_history_attn == 0).bool().to(self.model.device)
        sem_behavior_emb = self.transformer(history_emb, history_mask)

        target_emb = self.model.shared(target_item).view(batch_size, -1, self.model.config.hidden_size).to(self.model.device)
        sem_attn, target_emb = self.attn_pooling(sem_behavior_emb, target_emb, history_mask, target_item_attn)

        return sem_attn, target_emb

class Target_Attention_weight(nn.Module):
    def __init__(self, hid_dim1, hid_dim2):
        super().__init__()

        self.W = nn.Parameter(torch.randn((1, hid_dim1, hid_dim2)))
        nn.init.xavier_normal_(self.W)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq_emb, target, mask, target_attn):
        score = torch.matmul(seq_emb, self.W)
        seq_len = target_attn.sum(1, keepdim=True)
        target = target.masked_fill(target_attn.unsqueeze(2), 0)
        sum_emb = target.sum(dim=1)
        mean_emb = sum_emb / seq_len
        mean_emb = mean_emb.masked_fill(seq_len == 0, 0)
        # target = target.mean(dim=1) 
        score = torch.matmul(score, mean_emb.unsqueeze(-1))
        mask = mask.bool()

        all_score = score.masked_fill(mask.unsqueeze(-1), torch.tensor(-1e16))
        all_weight = self.softmax(all_score.transpose(-2, -1))

        return all_weight
