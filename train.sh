#!/bin/bash

/home/kvilasri/.cache/pypoetry/virtualenvs/unbabel-comet-UbVbjqzU-py3.10/bin/python \
    ./comet/cli/train.py \
    --cfg configs/models/referenceless_model.yaml \
    --load_from_checkpoint checkpoints/wmt21-comet-qe-mqm/checkpoints/model.ckpt
