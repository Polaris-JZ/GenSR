import logging

def parse_global_args(parser):
    # global args
    parser.add_argument("--token", type=str, default="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", help="Token")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument('--logging_level', type=int, default=logging.INFO,help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--phase', type=str, default='train', help='Phase')
    parser.add_argument('--base_print_interval', type=int, default=100, help='Print interval')

    # distributed learning
    parser.add_argument("--dist_url", type=str, default="env://", help="Distributed URL")
    parser.add_argument("--master_addr", type=str, default='localhost', help='Setup MASTER_ADDR for os.environ')
    parser.add_argument("--master_port", type=str, default='12342', help='Setup MASTER_PORT for os.environ')

    parser.add_argument("--base_seq_length", type=int, default=512, help="Base sequence length")
    parser.add_argument("--base_warmup_steps", type=int, default=200, help="Warm up steps")
    parser.add_argument("--base_max_steps", type=int, default=60, help="Max steps")
    parser.add_argument("--base_iter_per_epoch", type=int, default=50, help="Iter per epoch")

    parser.add_argument("--base_lr", type=float, default=1e-2, help="learning rate for base model")
    parser.add_argument("--base_min_lr", type=float, default=8e-5, help="Min learning rate for base model")
    parser.add_argument("--base_warmup_lr", type=float, default=1e-5, help="Warm up learning rate for base model")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Clip grad norm for base model")
    parser.add_argument("--accumulate_steps", type=int, default=8, help="Accumulate grad batches")

    parser.add_argument("--base_weight_decay", type=float, default=1e-3, help="Weight decay for base model")
    parser.add_argument("--base_lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type for base model")
    parser.add_argument("--base_num_epochs", type=int, default=200, help="Number of epochs for base model")
    parser.add_argument("--base_train_batch_size", type=int, default=16, help="Batch size for base model")
    parser.add_argument("--base_valid_batch_size", type=int, default=32, help="Batch size for base model")
    
    # eval
    parser.add_argument("--base_eval_times_per_epoch", type=int, default=2, help="Evaluation times per epoch")
    parser.add_argument("--base_early_stop", type=int, default=5, help="Early stop step")
    parser.add_argument("--topk", type=list, default=[5,10,20], help="Top k")
    parser.add_argument("--metrics", type=list, default=["HR", "NDCG"], help="Metrics")
    parser.add_argument("--main_metric", type=str, default="HR@5", help="Main metric")

    # data
    parser.add_argument("--train_ratio", type=float, default=1, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float, default=0.5, help="Valid ratio")
    parser.add_argument("--max_his_len", type=int, default=20, help="Max history length")

    # model
    parser.add_argument("--item_emb_path", type=str, default="./dataset/lgn-gowalla-2-64.pth.tar", help="Item embedding path")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Base model name")
    parser.add_argument("--model_ckpt_path", type=str, default="./outputs/best_model.pt", help="Model path")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--cf_emb_size", type=int, default=64, help="CF embedding size")
    parser.add_argument("--filter_threshold", type=float, default=0.4, help="Filter threshold")
    parser.add_argument("--filter_item_num", type=int, default=5, help="Filter item number")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12, help="Layer norm epsilon")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="Hidden dropout probability")
    parser.add_argument("--contrastive_weight", type=float, default=0.1, help="Contrastive weight")
    return parser

    