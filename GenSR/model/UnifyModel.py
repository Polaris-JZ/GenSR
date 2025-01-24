import torch
import logging
import time

import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from model.runner import maybe_autocast
from model.CFModel import CFModel
from model.SemModel import SemModel
class UnifyBaseModel(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        # define the model and tokenizer
        self.args = args
        self.contrastive_weight = args.contrastive_weight
        self.tokenizer = tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name)
        self.model.to(args.device)
        self.cf_model = CFModel(args)
        self.cf_model.to(args.device)
        self.sem_model = SemModel(args, self.model, self.tokenizer)
        self.sem_model.to(args.device)
        
        self.item_projection = nn.Linear(self.cf_model.get_cf_emb().shape[1], self.model.config.hidden_size)
        self.id_layer_norm = nn.LayerNorm(self.model.config.hidden_size).to(args.device)
        self.sem_layer_norm = nn.LayerNorm(self.model.config.hidden_size).to(args.device)
    def forward(self, batch, **kwargs):
        bs = batch['first_inputs_id'].shape[0]
        
        first_embeds = self.model.shared(batch['first_inputs_id'].to(self.model.device))
        second_embeds = self.model.shared(batch['second_inputs_id'].to(self.model.device))
        third_embeds = self.model.shared(batch['third_inputs_id'].to(self.model.device))
        fourth_embeds = self.model.shared(batch['fourth_inputs_id'].to(self.model.device))
        fifth_embeds = self.model.shared(batch['fifth_inputs_id'].to(self.model.device))
        first_mask = batch['first_inputs_attn'].to(self.model.device)
        second_mask = batch['second_inputs_attn'].to(self.model.device)
        third_mask = batch['third_inputs_attn'].to(self.model.device)
        fourth_mask = batch['fourth_inputs_attn'].to(self.model.device)
        fifth_mask = batch['fifth_inputs_attn'].to(self.model.device)

        # project batch['item_list'] and batch['target_item'] to model hidden size
        target_item_emb = self.item_projection(self.cf_model.item_emb_layer(batch['target_item'].to(self.model.device)))
        item_list_emb = self.item_projection(self.cf_model(batch['item_list'].to(self.model.device), batch['target_item'].to(self.model.device))).unsqueeze(1)

        # get the filtered semantic_item_list
        sem_item_emb, target_sem_emb = self.sem_model(batch['target_item_title'], batch['target_item_title_attn'], batch['history_item_title_id'], batch['history_item_title_attn'], batch['item_num'])
        # print(f'sem_item_emb: {sem_item_emb.shape}, target_sem_emb: {target_sem_emb.shape}')
        # sem_item_emb = sem_item_emb.unsqueeze(1)
        # target_sem_emb = target_sem_emb.unsqueeze(1)

        # calculate the contrastive loss
        # print(f'item_list_emb: {item_list_emb.shape}, sem_item_emb: {sem_item_emb.shape}')
        contrastive_loss = self.id_sem_loss(item_list_emb.squeeze(1), sem_item_emb.squeeze(1))
        
        
        # tokenize the filtered_semantic_item_list
        # filtered_semantic_item_list_token = self.tokenizer(filtered_semantic_item_list, padding="longest", truncation=True, max_length=512, return_tensors='pt', add_special_tokens=False)


        # create a item_list_mask (bs, 1), all 1
        item_list_mask = torch.ones([bs, 1]).to(self.model.device)  

        # concat first_embeds, filtered_semantic_item_list_emb, second_embeds, item_list_emb, third_embeds, target_item_emb, fourth_embeds
        inputs = torch.cat([first_embeds, item_list_emb, second_embeds, sem_item_emb, third_embeds, target_item_emb, fourth_embeds, target_sem_emb, fifth_embeds], dim=1)
        attention_mask = torch.cat([first_mask, item_list_mask, second_mask, item_list_mask, third_mask, item_list_mask, fourth_mask, item_list_mask, fifth_mask], dim=1)

        # get the target embs and attn
        target = batch['output_id'].to(self.model.device)

        # concat inputs and target_embeds
        inputs_embeds = inputs
        attention_mask = attention_mask
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=target,
            return_dict=True,
            **kwargs
        )

        loss = outputs["loss"]
        overall_loss = loss + self.contrastive_weight * contrastive_loss
        loss_dict = {'loss': loss, 'contrastive_loss': contrastive_loss}
        return overall_loss, loss_dict

    def id_sem_loss(self, id_emb, sem_emb):
        '''
        Input:
        id_emb: (batch_size, emb_size)
        sem_emb: (batch_size, emb_size)
        Output: id_sem_loss
        Function: calculate the contrastive learning loss between id and sem, use the other samples in the batch as negative samples
        '''
        
        # Normalize embeddings
        id_emb = nn.functional.normalize(id_emb, dim=1)
        sem_emb = nn.functional.normalize(sem_emb, dim=1)
        
        # 可选：添加额外的归一化层
        id_emb = self.id_layer_norm(id_emb)
        sem_emb = self.sem_layer_norm(sem_emb)

        # Calculate similarity matrix
        sim_matrix = torch.matmul(id_emb, sem_emb.T)  # (batch_size, batch_size)
        
        # Temperature scaling
        sim_matrix = sim_matrix / 0.07  # Using 0.07 as default temperature
        
        # Labels are on the diagonal (positive pairs)
        labels = torch.arange(id_emb.shape[0], device=id_emb.device)
        
        # Calculate loss in both directions (id->sem and sem->id)
        loss_id_sem = nn.functional.cross_entropy(sim_matrix, labels)
        loss_sem_id = nn.functional.cross_entropy(sim_matrix.T, labels)
        
        # Take average of both directions
        loss = (loss_id_sem + loss_sem_id) / 2
        
        return loss
      
    def get_cf_emb(self):
        # load item emb .tar file
        cf_emb = torch.load(self.args.item_emb_path)
        item_cf_emb = cf_emb['embedding_item.weight'].cpu().detach()
        item_cf_emb = torch.cat([torch.zeros(1, 64), item_cf_emb], 0)
        return item_cf_emb
    
    def generate(self, batch, **kwargs):
        bs = batch['first_inputs_id'].shape[0]
        
        first_embeds = self.model.shared(batch['first_inputs_id'].to(self.model.device))
        second_embeds = self.model.shared(batch['second_inputs_id'].to(self.model.device))
        third_embeds = self.model.shared(batch['third_inputs_id'].to(self.model.device))
        fourth_embeds = self.model.shared(batch['fourth_inputs_id'].to(self.model.device))
        fifth_embeds = self.model.shared(batch['fifth_inputs_id'].to(self.model.device))
        first_mask = batch['first_inputs_attn'].to(self.model.device)
        second_mask = batch['second_inputs_attn'].to(self.model.device)
        third_mask = batch['third_inputs_attn'].to(self.model.device)
        fourth_mask = batch['fourth_inputs_attn'].to(self.model.device)
        fifth_mask = batch['fifth_inputs_attn'].to(self.model.device)

        # project batch['item_list'] and batch['target_item'] to model hidden size
        target_item_emb = self.item_projection(self.cf_model.item_emb_layer(batch['target_item'].to(self.model.device)))
        item_list_emb = self.item_projection(self.cf_model(batch['item_list'].to(self.model.device), batch['target_item'].to(self.model.device))).unsqueeze(1)

        # get the filtered semantic_item_list
        sem_item_emb, target_sem_emb = self.sem_model(batch['target_item_title'], batch['target_item_title_attn'], batch['history_item_title_id'], batch['history_item_title_attn'], batch['item_num'])
        # print(f'sem_item_emb: {sem_item_emb.shape}, target_sem_emb: {target_sem_emb.shape}')
        # sem_item_emb = sem_item_emb.unsqueeze(1)
        # target_sem_emb = target_sem_emb.unsqueeze(1)
        
        # tokenize the filtered_semantic_item_list
        # filtered_semantic_item_list_token = self.tokenizer(filtered_semantic_item_list, padding="longest", truncation=True, max_length=512, return_tensors='pt', add_special_tokens=False)


        # create a item_list_mask (bs, 1), all 1
        item_list_mask = torch.ones([bs, 1]).to(self.model.device)  

        # concat first_embeds, filtered_semantic_item_list_emb, second_embeds, item_list_emb, third_embeds, target_item_emb, fourth_embeds
        inputs = torch.cat([first_embeds, item_list_emb, second_embeds, sem_item_emb, third_embeds, target_item_emb, fourth_embeds, target_sem_emb, fifth_embeds], dim=1)
        attention_mask = torch.cat([first_mask, item_list_mask, second_mask, item_list_mask, third_mask, item_list_mask, fourth_mask, item_list_mask, fifth_mask], dim=1)

        outputs = self.model.generate(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            max_new_tokens=1,
            output_scores=True,
            **kwargs
        )

        scores = outputs.scores[0]
        self.pos_ans_id = self.tokenizer('Yes',add_special_tokens=False).input_ids[0]
        self.neg_ans_id = self.tokenizer('No',add_special_tokens=False).input_ids[0]
        yes_prob = F.softmax(scores, dim=-1)[:, self.pos_ans_id]
        return yes_prob