from typing import Any, Dict, List
from tqdm import tqdm

import numpy as np
import torch.optim
from torch.utils.data import DataLoader
import torch
import logging

import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Subset
from utils import parser, utils, eval
from model import runner
from transformers import EarlyStoppingCallback
from data.UnifyDataset import UnifyDataset
from data.UnifyCollator import UnifyCollator
from data.UnifySampler import CustomDistributedSampler
from model.UnifyModel import UnifyBaseModel
from torch.utils.data import DataLoader

def evaluate_method(predictions: np.ndarray, topk: int,
                    metrics: List[str]) -> Dict[str, float]:
    evaluations = dict()
    sort_idx = (-predictions).argsort(axis=1)
    gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
    for k in topk:
        hit = (gt_rank <= k)
        for metric in metrics:
            key = '{}@{}'.format(metric, k)
            if metric == 'HR':
                evaluations[key] = hit.mean()
            elif metric == 'NDCG':
                evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
            else:
                raise ValueError(
                    'Undefined evaluation metric: {}.'.format(metric))
    return evaluations

def format_metric(result_dict: Dict[str, Any]) -> str:
    assert type(result_dict) == dict
    format_str = []
    metrics = np.unique([k.split('@')[0] for k in result_dict.keys()])
    topks = np.unique([int(k.split('@')[1]) for k in result_dict.keys()])
    for topk in np.sort(topks):
        for metric in np.sort(metrics):
            name = '{}@{}'.format(metric, topk)
            m = result_dict[name]
            if type(m) is float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('{}:{:<.4f}'.format(name, m))
            elif type(m) is int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('{}:{}'.format(name, m))
    return ','.join(format_str)

def test(args, tokenizer):
    base_model = UnifyBaseModel(args, tokenizer)
    base_model.to(args.device)
    if args.rank == 0:
        logging.info(f"Loaded the base_model: {args.base_model_name}")

    # load the model
    base_model.load_state_dict(torch.load(args.model_ckpt_path))

    # prepare DDP
    base_model = DDP(base_model, device_ids=[args.rank], find_unused_parameters=True)


    # load the dataset
    test_rec_dataset = UnifyDataset("test", args, task='rec')
    test_src_dataset = UnifyDataset("test", args, task='src')
    if args.rank == 0:
        logging.info(f"Loaded the test dataset")
    collator = UnifyCollator(tokenizer, args)

    # get the distributed sampler
    test_rec_sampler = CustomDistributedSampler(test_rec_dataset, args.world_size, args.rank)
    test_src_sampler = CustomDistributedSampler(test_src_dataset, args.world_size, args.rank)

    # get the dataloader
    test_rec_dataloader = DataLoader(test_rec_dataset, batch_size=args.base_valid_batch_size, sampler=test_rec_sampler, collate_fn=collator)
    test_src_dataloader = DataLoader(test_src_dataset, batch_size=args.base_valid_batch_size, sampler=test_src_sampler, collate_fn=collator)
    # base_model = UnifyBaseModel(args, tokenizer)
    # base_model = base_model.to(args.device_id)
    # if args.rank == 0:
    #     logging.info(f"Loaded the base_model: {args.base_model_name}")

    # # load the model
    # base_model.load_state_dict(torch.load(args.model_ckpt_path))

    # # prepare DDP
    # base_model = DDP(base_model, device_ids=[args.device_id], find_unused_parameters=True)

    # load the dataset
    # test_rec_dataset = UnifyDataset("test", args, task='rec')
    # test_src_dataset = UnifyDataset("test", args, task='src')
    # if args.rank == 0:
    #     logging.info(f"Loaded the test dataset")
    # collator = UnifyCollator(tokenizer, args)

    # # get the distributed sampler
    # test_rec_sampler = CustomDistributedSampler(test_rec_dataset, args.world_size, args.rank)
    # test_src_sampler = CustomDistributedSampler(test_src_dataset, args.world_size, args.rank)

    # # get the dataloader
    # test_rec_dataloader = DataLoader(test_rec_dataset, batch_size=100, sampler=test_rec_sampler, collate_fn=collator)
    # test_src_dataloader = DataLoader(test_src_dataset, batch_size=100, sampler=test_src_sampler, collate_fn=collator)

    base_model.eval()
    with torch.no_grad():
        predictions = list()
        cnt = 0
        for batch in tqdm(test_rec_dataloader):
            cnt += 1
            # move batch to device
            for key in batch:
                batch[key] = batch[key].to(args.device)
            logit = base_model.module.generate(batch)
            for i in range(args.base_valid_batch_size//100):
                logit_part = logit[i*100:(i+1)*100]
                if len(logit_part) != 0:
                    predictions.append(logit_part.cpu().data.numpy())
        dist.barrier()
        # gather the predictions from all GPUs
        all_predictions = [None for _ in range(dist.get_world_size())]
        # print(predictions)
        dist.all_gather_object(all_predictions, predictions)
        # flatten the list of predictions
        predictions = [item for sublist in all_predictions for item in sublist]
        # print(len(predictions))
        # print(predictions)
        predictions = np.array(predictions)
        valid_results = eval.evaluate_method(predictions, args.topk, args.metrics)
        valid_metric = eval.format_metric(valid_results)
        main_result = valid_results[args.main_metric]
        if args.rank == 0:
            logging.info(f"[Test] Evaluation Results for rec: {valid_metric}")
    dist.barrier()

    base_model.eval()
    with torch.no_grad():
        predictions = list()
        cnt = 0
        for batch in tqdm(test_src_dataloader):
            cnt += 1
            # move batch to device
            for key in batch:
                batch[key] = batch[key].to(args.device)
            logit = base_model.module.generate(batch)
            for i in range(args.base_valid_batch_size//100):
                logit_part = logit[i*100:(i+1)*100]
                if len(logit_part) != 0:
                    predictions.append(logit_part.cpu().data.numpy())
        dist.barrier()
        # gather the predictions from all GPUs
        all_predictions = [None for _ in range(dist.get_world_size())]
        # print(predictions)
        dist.all_gather_object(all_predictions, predictions)
        # flatten the list of predictions
        predictions = [item for sublist in all_predictions for item in sublist]
        # print(len(predictions))
        # print(predictions)
        predictions = np.array(predictions)
        valid_results = eval.evaluate_method(predictions, args.topk, args.metrics)
        valid_metric = eval.format_metric(valid_results)
        main_result = valid_results[args.main_metric]
        if args.rank == 0:
            logging.info(f"[Test] Evaluation Results for src: {valid_metric}")
    dist.barrier()