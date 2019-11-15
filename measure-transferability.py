from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np

from pgd_attack import LinfPGDAttack


parser = argparse.ArgumentParser(description='TF MNIST')
parser.add_argument('--base-dir', default='/home/hzzheng/Code/madryDefense/mnist_challenge_transferability/')
parser.add_argument('--target-model', default='data-model/model_r10000/checkpoint-99900',
                    help='Log path.')
parser.add_argument('--path', default='data-adv/model.r10000.checkpoint-99900.A.adv.npy',
                    help='Log path.')
parser.add_argument('--gpuid', type=int, default=0,
                    help='The ID of GPU.')

args = parser.parse_args()

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

# log_file = open(args.log_path, 'w')

if __name__ == '__main__':
    import json
    import sys
    import math

    from tensorflow.examples.tutorials.mnist import input_data

    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_file = args.base_dir + args.target_model
    data_path = args.base_dir + args.path

    if model_file is None:
        print('No model found')
        sys.exit()

    model = Model()
    # attack = LinfPGDAttack(model,
    #                        config['epsilon'],
    #                        config['k'],
    #                        config['a'],
    #                        config['random_start'],
    #                        config['loss_func'])
    saver = tf.train.Saver()

    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    with tf.Session() as sess:
        # Restore the checkpoint
        saver.restore(sess, model_file)

        # Iterate over the samples batch-by-batch
        num_eval_examples = config['num_eval_examples']
        eval_batch_size = config['eval_batch_size']
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        x_adv = np.load(data_path)

        # print('Iterating over {} batches'.format(num_batches))

        total_nat_corr = 0
        total_adv_corr = 0
        total_adv_loss = 0
        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            # print('batch size: {}'.format(bend - bstart))

            x_batch = mnist.test.images[bstart:bend, :]
            y_batch = mnist.test.labels[bstart:bend]

            x_batch_adv = x_adv[bstart:bend]

            nat_dict = {model.x_input: x_batch,
                            model.y_input: y_batch}
            adv_dict = {model.x_input: x_batch_adv,
                            model.y_input: y_batch}

            nat_corr = sess.run(model.num_correct, feed_dict=nat_dict)
            adv_corr, adv_loss = sess.run([model.num_correct, model.xent], feed_dict=adv_dict)

            total_adv_loss += adv_loss
            total_nat_corr += nat_corr
            total_adv_corr += adv_corr
        
        total_adv_loss /= num_eval_examples
        total_nat_corr = total_nat_corr / num_eval_examples * 100
        total_adv_corr = total_adv_corr / num_eval_examples * 100
        print("Nat Acc:{} Adv Acc: {} Adv Loss: {}".format(total_nat_corr, total_adv_corr, total_adv_loss))
