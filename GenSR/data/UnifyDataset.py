import pickle
import os
import random

from torch.utils.data import Dataset, DataLoader

class UnifyDataset(Dataset):
    def __init__(self, mode, args, task='rec'):
        super().__init__()
        self.mode = mode
        self.args = args
        self.eval_task = task
        self.task_dict = {'rec': 0, 'src': 1}

        self.prompt = self.get_prompt()
        self.item_meta = self.get_item_meta()
        self.ori_item_meta = self.load_ori_item_data()

        self.load_dataset(mode)

        

    def load_ori_item_data(self):
        """加载商品数据文件"""
        file_path = './dataset/item_plain_text.txt'
        item_text_dict = {}
        with open(file_path, 'r') as file:
            for line in file:
                id_, text = line.split(' ', 1)
                item_text_dict[int(id_)] = text.strip()  
        return item_text_dict

    def get_item_meta(self):
        item_text_file = './dataset/item_plain_text.pkl'
        with open(item_text_file, 'rb') as f:
            item_text_dict = pickle.load(f)
        return item_text_dict


    def load_dataset(self, mode):
        if mode == "train":
            data_path = './dataset/train.pkl'
        elif mode == "valid":
            data_path = './dataset/valid.pkl'
        elif mode == "test" and self.eval_task == "rec":
            data_path = './dataset/test_rec.pkl'
        elif mode == "test" and self.eval_task == "src":
            data_path = './dataset/test_src.pkl'
        else:
            raise ValueError("Invalid mode: {}".format(mode))

        with open(data_path, 'rb') as f:
            ori_data = pickle.load(f)

        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        self.data['item_list'] = []
        self.data['target_item'] = []
        self.data['task'] = []
        self.data['query'] = []
        self.data['target_item_title'] = []
        self.data['history_item_title'] = []
        if mode == "valid" and self.args.valid_ratio < 1:
            random_indices = random.sample(range(len(ori_data)), int(len(ori_data) * self.args.valid_ratio))
            new_data = {}
            for index, data in ori_data.items():
                if index in random_indices:
                    new_data[index] = data
                else:
                    continue
            ori_data = new_data
        for index, data in ori_data.items():
            task = data['task']
            if mode == 'train':
                sample_num = len(data['neg_items']) + 1
                target_item_list = [data['target_item']] + data['neg_items']
                output_list = ['Yes'] + ['No'] * len(data['neg_items'])
            elif mode == 'valid' or mode == 'test':
                sample_num = len(data['neg_items'])
                target_item_list = [data['target_item']] + data['neg_items'][:-1]
                output_list = ['Yes'] + ['No'] * len(data['neg_items'])
            for i in range(sample_num):
                if task == 'rec':
                    # only get the first 10 of each item_id, if less than 10, pad with 0
                    item_title_list = [self.item_meta[item_id]['input_ids'][:10] + [0] * (10 - len(self.item_meta[item_id]['input_ids'])) if len(self.item_meta[item_id]['input_ids']) < 10 else self.item_meta[item_id]['input_ids'][:10] for item_id in data['item_list']]
                    self.data['history_item_title'].append(item_title_list)
                    # chenge list to string (seperate by ';')
                    # item_title_list = ';'.join(item_title_list)
                    target_item_title = self.ori_item_meta[target_item_list[i]]
                    self.data['input'].append(self.prompt[task])
                    self.data['output'].append(output_list[i])
                    self.data['item_list'].append(data['item_list'])
                    self.data['target_item'].append([target_item_list[i]])
                    self.data['task'].append(self.task_dict[task])
                    self.data['query'].append(data['query'][-1])
                    self.data['target_item_title'].append(target_item_title)
                elif task == 'src':
                    item_title_list = [self.item_meta[item_id]['input_ids'][:10] + [0] * (10 - len(self.item_meta[item_id]['input_ids'])) if len(self.item_meta[item_id]['input_ids']) < 10 else self.item_meta[item_id]['input_ids'][:10] for item_id in data['item_list']]
                    self.data['history_item_title'].append(item_title_list)
                    # chenge list to string (seperate by ';')
                    # item_title_list = ';'.join(item_title_list)                   
                    target_item_title = self.ori_item_meta[target_item_list[i]]
                    self.data['input'].append(self.prompt[task].format(**{'Query': data['query'][-1]}))
                    self.data['output'].append(output_list[i])
                    self.data['item_list'].append(data['item_list'])
                    self.data['target_item'].append([target_item_list[i]])
                    self.data['task'].append(self.task_dict[task])
                    self.data['query'].append(data['query'][-1])
                    self.data['target_item_title'].append(target_item_title)
        

    def get_prompt(self):
        prompt = {}
        prompt['rec'] = """ 
                    ### Question: 
                    A user has given high ratings to the following books: <ItemTitleList>. The semantic features of these books are <ItemFeatureList>.
                    Using all available information, make a prediction about whether the user would enjoy the book <TargetItemTitle> with the semantic feature <TargetItemID>? 
                    Answer with "Yes" or "No". 
                    \n #Answer:
                """
        prompt['src'] = """ 
                    ### Question: 
                    A user has given high ratings to the following books: <ItemTitleList>. The semantic features of these books are <ItemFeatureList>.
                    Using all available information, make a prediction about whether the user would enjoy the book <TargetItemTitle> with the semantic feature <TargetItemID> based on the query <Query>? 
                    Answer with "Yes" or "No". 
                    \n #Answer:
                """
        return prompt

    def __getitem__(self, idx):
        
        return {'input': self.data['input'][idx],
                'output': self.data['output'][idx],
                'item_list': self.data['item_list'][idx],
                'target_item': self.data['target_item'][idx],
                'query': self.data['query'][idx],
                'target_item_title': self.data['target_item_title'][idx],
                'history_item_title': self.data['history_item_title'][idx]}

    def __len__(self):
        return len(self.data['input'])
    
