# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Evaluate model on dev set with full metrics (AUC, MRR, nDCG5, nDCG10).
Usage:
    CUDA_VISIBLE_DEVICES=0 python evaluate_dev.py \
        --pretrained_model_path ./models/bert-tiny \
        --root_data_dir ./data/speedy_data/ \
        --load_ckpt_name ./saved_models/fastformer_final_test-epoch-1.pt \
        --batch_size 256 \
        --news_dim 256
"""

import os
import logging
import torch
import numpy as np

from utility.utils import setuplogger, check_args_environment
from utility.metrics import MetricsDict
from data_handler.preprocess import get_news_feature, infer_news
from data_handler.TestDataloader import DataLoaderTest
from models.speedyrec import MLNR


def evaluate_on_dev(args):
    setuplogger()
    args = check_args_environment(args)
    logging.info('-----------start evaluation on dev------------')

    local_rank = 0
    if args.enable_gpu:
        device = torch.device('cuda', int(local_rank))
    else:
        device = torch.device('cpu')

    model = MLNR(args)
    model = model.to(device)
    
    ckpt = torch.load(args.load_ckpt_name, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    logging.info(f"Loaded checkpoint from {args.load_ckpt_name}")

    auc = test_dev(model, args, device, ckpt['category_dict'], ckpt['subcategory_dict'])
    logging.info(f"Final AUC on dev set: {auc:.4f}")
    return auc


def test_dev(model, args, device, category_dict, subcategory_dict):
    model.eval()

    with torch.no_grad():
        # Load dev data (not test)
        news_info, news_combined = get_news_feature(
            args, mode='dev', 
            category_dict=category_dict,
            subcategory_dict=subcategory_dict
        )
        news_vecs = infer_news(model, device, news_combined)

        dataloader = DataLoaderTest(
            news_index=news_info.news_index,
            news_scoring=news_vecs,
            data_dirs=[os.path.join(args.root_data_dir, 'dev/')],
            filename_pat=args.filename_pat,
            args=args,
            world_size=1,
            worker_rank=0,
            cuda_device_idx=0,
            enable_prefetch=args.enable_prefetch,
            enable_shuffle=False,
            enable_gpu=args.enable_gpu,
        )

        results = MetricsDict(metrics_name=["AUC", "MRR", "nDCG5", "nDCG10"])
        results.add_metric_dict('all users')
        results.add_metric_dict('cold users')

        cnt = 0
        for cnt, (log_vecs, log_mask, news_vecs, labels) in enumerate(dataloader):
            his_lens = torch.sum(log_mask, dim=-1).to(torch.device("cpu")).detach().numpy()

            if args.enable_gpu:
                log_vecs = log_vecs.cuda(device=device, non_blocking=True)
                log_mask = log_mask.cuda(device=device, non_blocking=True)

            user_vecs = model.user_encoder(
                log_vecs, log_mask, user_log_mask=True).to(torch.device("cpu")).detach().numpy()

            for index, user_vec, news_vec, label, his_len in zip(
                    range(len(labels)), user_vecs, news_vecs, labels, his_lens):

                if label.mean() == 0 or label.mean() == 1:
                    continue
                score = np.dot(news_vec, user_vec)

                metric_rslt = results.cal_metrics(score, label)
                results.update_metric_dict('all users', metric_rslt)

                if his_len <= 5:
                    results.update_metric_dict('cold users', metric_rslt)

            if cnt % 100 == 0:
                results.print_metrics(0, cnt * args.batch_size, 'all users')

        dataloader.join()
        
        logging.info("=" * 50)
        logging.info("FINAL RESULTS ON DEV SET:")
        results.print_metrics(0, cnt * args.batch_size, 'all users')
        results.print_metrics(0, cnt * args.batch_size, 'cold users')
        logging.info("=" * 50)

    return np.mean(results.metrics_dict["all users"]['AUC'])


if __name__ == "__main__":
    from parameters import parse_args
    setuplogger()
    args = parse_args()
    evaluate_on_dev(args)
