import torch
import numpy as np
import time
import logging
class UnifyCollator:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        ori_input = [item['input'] for item in batch]
        ori_input_first = [input_text.split("<ItemTitleList>")[0] for input_text in ori_input]
        ori_input_second = [input_text.split("<ItemTitleList>")[1] for input_text in ori_input]
        ori_input_second = [input_text.split("<ItemFeatureList>")[0] for input_text in ori_input_second]
        ori_input_third = [input_text.split("<ItemFeatureList>")[1] for input_text in ori_input]
        ori_input_third = [input_text.split("<TargetItemTitle>")[1] for input_text in ori_input_third]
        ori_input_fourth = [input_text.split("<TargetItemTitle>")[1] for input_text in ori_input]
        ori_input_fourth = [input_text.split("<TargetItemID>")[0] for input_text in ori_input_fourth]
        ori_input_fifth = [input_text.split("<TargetItemID>")[1] for input_text in ori_input]

        first_inputs_id, first_inputs_attn = self.tokenizer.batch_encode_plus(
            ori_input_first, padding="longest", truncation=True, max_length=512, return_tensors='pt', add_special_tokens=False
        ).values()
        second_inputs_id, second_inputs_attn = self.tokenizer.batch_encode_plus(
            ori_input_second, padding="longest", truncation=True, max_length=512, return_tensors='pt', add_special_tokens=False
        ).values()
        third_inputs_id, third_inputs_attn = self.tokenizer.batch_encode_plus(
            ori_input_third, padding="longest", truncation=True, max_length=512, return_tensors='pt', add_special_tokens=False
        ).values()
        fourth_inputs_id, fourth_inputs_attn = self.tokenizer.batch_encode_plus(
            ori_input_fourth, padding="longest", truncation=True, max_length=32, return_tensors='pt', add_special_tokens=False
        ).values()
        fifth_inputs_id, fifth_inputs_attn = self.tokenizer.batch_encode_plus(
            ori_input_fifth, padding="longest", truncation=True, max_length=32, return_tensors='pt', add_special_tokens=False
        ).values()

        item_list = [item['item_list'] for item in batch]
        max_len = self.args.max_his_len
        pad_item_list = [item + [0] * (max_len - len(item)) for item in item_list]
        item_list_mask = [[1] * len(item) + [0] * (max_len - len(item)) for item in item_list]
        
        target_item = [item['target_item'] for item in batch]
        target_item_mask = [[1] for item in target_item]

        output = [item['output'] for item in batch]
        output_id, output_attn = self.tokenizer.batch_encode_plus(
            output, padding="longest", truncation=True, max_length=32, return_tensors='pt', add_special_tokens=False
        ).values()

        label_number = [1 if item == 'Yes' else 0 for item in output]
        query = [str(item['query']) for item in batch]
        history_item_title_id = [
            item['history_item_title'] + [[0] * 10 for _ in range(max_len - len(item['history_item_title']))]
            for item in batch
        ]
   
        history_item_title_id = torch.tensor(history_item_title_id).view(len(batch), -1)
        history_item_title_attn = (history_item_title_id != 0)

        query_id, query_attn = self.tokenizer.batch_encode_plus(
            query, padding="longest", truncation=True, max_length=32, return_tensors='pt', add_special_tokens=False
        ).values()

        target_item_title = [item['target_item_title'] for item in batch]

        target_item_title_id, target_item_title_attn = self.tokenizer.batch_encode_plus(
            target_item_title, padding="longest", truncation=True, max_length=32, return_tensors='pt', add_special_tokens=False
        ).values()

        item_num = [len(item['item_list']) for item in batch]
        result = {
            'first_inputs_id': torch.tensor(first_inputs_id),
            'first_inputs_attn': torch.tensor(first_inputs_attn),
            'second_inputs_id': torch.tensor(second_inputs_id),
            'second_inputs_attn': torch.tensor(second_inputs_attn),
            'third_inputs_id': torch.tensor(third_inputs_id),
            'third_inputs_attn': torch.tensor(third_inputs_attn),
            'fourth_inputs_id': torch.tensor(fourth_inputs_id),
            'fourth_inputs_attn': torch.tensor(fourth_inputs_attn),
            'fifth_inputs_id': torch.tensor(fifth_inputs_id),
            'fifth_inputs_attn': torch.tensor(fifth_inputs_attn),
            'item_list': torch.tensor(pad_item_list),
            'target_item': torch.tensor(target_item),
            'item_list_mask': torch.tensor(item_list_mask),
            'target_item_mask': torch.tensor(target_item_mask),
            'output_id': torch.tensor(output_id),
            'output_attn': torch.tensor(output_attn),
            'label_number': torch.tensor(label_number),
            'query':torch.tensor(query_id),
            'history_item_title_id': torch.tensor(history_item_title_id),
            'history_item_title_attn': torch.tensor(history_item_title_attn),
            'target_item_title': torch.tensor(target_item_title_id),
            'target_item_title_attn': torch.tensor(target_item_title_attn),
            'item_num': torch.tensor(item_num),
        }
        
        return result
  