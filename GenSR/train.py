import torch
import argparse
import os
import logging
import time
import random
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
from torch.utils.data import DistributedSampler
from transformers import get_linear_schedule_with_warmup

           
def main(args):
    # set device
    utils.init_distributed_mode(args)
    args.device = f"cuda:{args.rank}"
    
    # construct the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token_id = 0
    base_model = UnifyBaseModel(args, tokenizer)
    base_model.to(args.device)
    if args.rank == 0:
        logging.info(f"Loaded the base_model: {args.base_model_name}")

    # prepare DDP
    base_model = DDP(base_model, device_ids=[args.rank], find_unused_parameters=True)

    # load the dataset
    train_dataset = UnifyDataset("train", args)
    if args.rank == 0:
        logging.info(f"Loaded the traindataset")
    collator = UnifyCollator(tokenizer, args)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.base_train_batch_size, sampler=train_sampler, collate_fn=collator, num_workers=8)

    valid_dataset = UnifyDataset("valid", args)
    if args.rank == 0:
        logging.info(f"Loaded the validataset")
    valid_sampler = CustomDistributedSampler(valid_dataset, args.world_size, args.rank)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.base_valid_batch_size, sampler=valid_sampler, collate_fn=collator, num_workers=8)

    # set optimizer and lr scheduler
    optimizer = runner.optimizer(base_model, args)
    # calculate steps for one epoch under distributed training
    num_steps_per_epoch = len(train_dataloader)
    total_training_steps = args.base_num_epochs * num_steps_per_epoch
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.base_warmup_steps, num_training_steps=total_training_steps)
    dist.barrier()
    # start the training
    best_valid_result = -float('inf')
    early_stop_cnt = args.base_early_stop
    stop_training = False
    
    args.base_eval_interval = len(train_dataloader) // args.base_eval_times_per_epoch

    for epoch in range(args.base_num_epochs):
        if stop_training:
            break
        train_loss = 0
        start_time = time.time()
        
        for step, batch in enumerate(train_dataloader):
            base_model.train()
            dist.barrier()
            # move batch to device
            for key in batch:
                batch[key] = batch[key].to(args.device)
            loss, loss_dict = base_model.module(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.module.parameters(), args.clip_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            current_loss = loss / args.world_size

            dist.all_reduce(loss_dict['loss'], op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_dict['contrastive_loss'], op=dist.ReduceOp.SUM)
            loss_dict['loss'] = loss_dict['loss'] / args.world_size
            loss_dict['contrastive_loss'] = loss_dict['contrastive_loss'] / args.world_size

            

            # print the loss every args.base_print_interval steps
            if step % args.base_print_interval == 0 and args.rank == 0:
                logging.info(f"[Train] Epoch {epoch} [{step}/{len(train_dataloader)}]: Training loss {current_loss}, current lr {optimizer.param_groups[0]['lr']}, contrastive loss {loss_dict['contrastive_loss']}, task loss {loss_dict['loss']}")

            dist.barrier()
            # evaluate every certain steps
            if (step + 1) % args.base_eval_interval == 0 and epoch > 2:
            # if step == 400:
                base_model.eval()
                with torch.no_grad():
                    # 预计算总批次数和预分配数组
                    total_batches = len(valid_dataloader)
                    batch_predictions = np.zeros((total_batches * (args.base_valid_batch_size // 100), 100))
                    
                    for batch_idx, batch in enumerate(tqdm(valid_dataloader)):
                        # 移动到GPU并生成预测
                        batch = {k: v.to(args.device) for k, v in batch.items()}
                        logits = base_model.module.generate(batch)
                        
                        # 批量处理预测结果
                        logits = logits.reshape(-1, 100)  # 直接重塑为所需形状
                        start_idx = batch_idx * (args.base_valid_batch_size // 100)
                        end_idx = start_idx + len(logits)
                        batch_predictions[start_idx:end_idx] = logits.cpu().numpy()

                    dist.barrier()
                    # gather the predictions from all GPUs
                    all_predictions = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(all_predictions, batch_predictions)
                    # flatten the list of predictions
                    predictions = [item for sublist in all_predictions for item in sublist]
                    # print(len(predictions))
                    # print(predictions)
                    predictions = np.array(predictions)
                    valid_results = eval.evaluate_method(predictions, args.topk, args.metrics)
                    valid_metric = eval.format_metric(valid_results)
                    main_result = valid_results[args.main_metric]
                    if args.rank == 0:
                        logging.info(f"[Valid] Evaluation Results: {valid_metric}")
                dist.barrier()
                

                if main_result > best_valid_result:
                    best_valid_result = main_result
                    early_stop_cnt = args.base_early_stop
                    best_model_path = os.path.join(args.output_dir, f"best_model.pt")
                    os.makedirs(args.output_dir, exist_ok=True)
                    torch.save(base_model.module.state_dict(), best_model_path)
                    if args.rank == 0:
                        logging.info(f"Best model saved to {best_model_path}")
                    dist.barrier()                  
                    
                else:
                    early_stop_cnt -= 1
                    if early_stop_cnt <= 0 and epoch > 1:
                        stop_training = True
                        if args.rank == 0:
                            logging.info(f"Early stopping at epoch {epoch}")
                        break
            
        train_loss /= len(train_dataloader)
        if args.rank == 0:
            logging.info(f"[Train] Epoch {epoch}: Overall Training loss {train_loss}")
            # change to hours
            hours = (time.time() - start_time) / 3600
            logging.info(f"[Time] Epoch {epoch}: Time taken {hours:.2f} hours")

    eval.test(args, tokenizer)
    

if __name__ == "__main__":
    # set the parser
    parsers = argparse.ArgumentParser(description='Base_llama')
    parsers = parser.parse_global_args(parsers)
    args, extras = parsers.parse_known_args()

    # set log
    utils.setup_logging(args)
    
    # set all seeds
    utils.set_seed(args.seed)

    main(args)