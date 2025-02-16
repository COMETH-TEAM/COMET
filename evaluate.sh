#!/bin/bash

/home/kvilasri/.cache/pypoetry/virtualenvs/unbabel-comet-UbVbjqzU-py3.10/bin/python \
	./comet/cli/evaluate.py \
	--model_path checkpoints/wmt21-comet-qe-mqm/checkpoints/model.ckpt \
	--file_path data/data-1736919579996/test_wo_bunny_with_label.csv
