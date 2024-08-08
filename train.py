import os
import argparse
import pickle as pkl
import random
import torch
import math
import json
import string
import logging
import numpy as np

from collections import Counter, defaultdict

from transformers import GPT2Tokenizer, AutoTokenizer

from MultiICL.data import MultiICLData
from MultiICL.model import MultiICLModel
from utils.data import load_data

from torch.utils.checkpoint import checkpoint


def main(logger, args):
    #加载模型
    if args.gpt2.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    batch_size = args.batch_size
    max_length_per_example = 256
    max_length = 256
    if args.use_demonstrations:
        max_length = min(max_length * args.k, 1024)

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.batch_size, max_length, max_length_per_example))

    train_data = load_data(args.task, "train", args.k, seed=args.seed)

    train_counter = Counter()
    for dp in train_data:
        train_counter[dp["task"]] += 1
    if args.local_rank <= 0:
        for k, v in train_counter.items():
            logger.info("[Train] %s\t%d" % (k, v))
        logger.info("%s on %s (%d train)" % (args.method, args.task, len(train_counter)))
#111
    if args.init_checkpoint is not None:
        assert os.path.exists(args.init_checkpoint)

    ######### load tensorize data
    multiICL_data = MultiICLData(logger, tokenizer, args.method, args.use_demonstrations,
                               args.test_k, max_length, max_length_per_example,
                               do_tensorize=args.do_tensorize,
                               tensorize_dir=args.tensorize_dir,
                               n_process=args.n_process, n_gpu=args.n_gpu, local_rank=args.local_rank)
    multiICL_data.tensorize_for_training(train_data, keyword=args.task, seed=args.seed,
                                        use_random_english_words=args.use_random_english_words)

    if args.do_tensorize:
        return

    ######## actual training part

    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.train_seed)

    num_training_steps = args.num_training_steps
    save_period = 10 #每10步保存一下
    log_period = 10

    if args.no_masking:
        multiICL_data.tensorized_inputs["token_type_ids"] = torch.ones_like(multiICL_data.tensorized_inputs["input_ids"])
    multiICL_data.print_tensorized_example()

    logger.info(args.out_dir)

    if args.local_rank<=0 and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    multiicl_model = MultiICLModel(logger, args.out_dir, args.fp16, args.local_rank)
    multiicl_model.load(args.init_checkpoint, args.gpt2)

    multiicl_model.to_device()
    multiicl_model.setup_optimizer(args.optimization, num_training_steps, args.lr,
                                  args.weight_decay, args.warmup_steps)
    multiicl_model.parallel()
    multiicl_model.train()
    multiicl_model.do_train(multiICL_data, args.batch_size, num_training_steps, save_period, log_period)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_tensorize", default=False, action="store_true")
    parser.add_argument("--tensorize_dir", type=str, default="tensorized")
    parser.add_argument("--n_gpu", type=int, default=8)
    parser.add_argument("--n_process", type=int, default=40)

    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default="Task1")
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--test_k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_training_steps", type=int, default=100)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--no_masking", default=False, action="store_true")


    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--method", type=str, default="direct", choices=["direct", "channel"])
    parser.add_argument("--gpt2", type=str, default="gpt2-xl")

    parser.add_argument("--optimization", type=str, default="adamw")
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
