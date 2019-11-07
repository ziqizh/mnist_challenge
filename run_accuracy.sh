#!/bin/bash

#python measure-accuracy.py --model-name mat.pgd-1
#python measure-accuracy.py --model-name mat.pgd-1 --ckpt-step 300 --start-ckpt 3000 --max-ckpt 9000
#python draw_accuracy.py --model-name mat.pgd-1
#
#python measure-accuracy.py --model-name mat.pgd-1.new
#python draw_accuracy.py --model-name mat.pgd-1.new
#
#python measure-accuracy.py --model-name mat.pgd-5
#python draw_accuracy.py --model-name mat.pgd-5
#
#python measure-accuracy.py --model-name mat.pgd-10.new
#python draw_accuracy.py --model-name mat.pgd-10.new
#
#python measure-accuracy.py --model-name mat.pgd-20
python measure-accuracy.py --model-name mat.pgd-20 --ckpt-step 300 --start-ckpt 51000 --max-ckpt 57000
#python draw_accuracy.py --model-name mat.pgd-20

python measure-accuracy.py --model-name mat.pgd-40
python measure-accuracy.py --model-name mat.pgd-40 --ckpt-step 300 --start-ckpt 51000 --max-ckpt 57000
#python draw_accuracy.py --model-name mat.pgd-40




