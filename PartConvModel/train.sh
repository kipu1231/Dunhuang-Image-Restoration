#!/usr/bin/env bash
# train_mode == "w_pretrain": pre-trains model on Places2 and fine-tunes on grotto data
# train_mode == "wo_pretrain": trains model only on grotto data
python3 train.py --train_mode $1