#!/bin/bash

python measure-accuracy.py --model-name mat.pgd-1
python draw_accuracy.py --model-name mat.pgd-1

python measure-accuracy.py --model-name mat.pgd-1.new
python draw_accuracy.py --model-name mat.pgd-1.new

python measure-accuracy.py --model-name mat.pgd-5
python draw_accuracy.py --model-name mat.pgd-5

python measure-accuracy.py --model-name mat.pgd-10.new
python draw_accuracy.py --model-name mat.pgd-10.new

python measure-accuracy.py --model-name mat.pgd-20
python draw_accuracy.py --model-name mat.pgd-20

python measure-accuracy.py --model-name mat.pgd-40
python draw_accuracy.py --model-name mat.pgd-40




