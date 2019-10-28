"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

from pgd_attack import LinfPGDAttack

parser = argparse.ArgumentParser(description='TF CIFAR PGD')
parser.add_argument('--log-prefix', default='./data-log/measure-accuracy/',
                    help='Log path.')
parser.add_argument('--gpuid', type=int, default=0,
                    help='The ID of GPU.')
parser.add_argument('--model-name', default='mat.pgd-40',
                    help='ATTA attack step.')
parser.add_argument('--model-dir', default='./models/data-model/',
                    help='The dir of the saved model')
parser.add_argument('--ckpt', type=int, default=0,
                    help='checkpoint')
parser.add_argument('--ckpt-step', type=int, default=3000,
                    help='checkpoint')
parser.add_argument('--max-ckpt', type=int, default=99900,
                    help='checkpoint')
parser.add_argument('--batch-size', type=int, default=128,
                    help='checkpoint')
args = parser.parse_args()

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

# log_file = open(args.log_path, 'w')

if __name__ == '__main__':
    import json

    from tensorflow.examples.tutorials.mnist import input_data

    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_name = args.model_name
    model_dir = args.model_dir + model_name

    model = Model()
    attack = LinfPGDAttack(model,
                           config['epsilon'],
                           config['k'],
                           config['a'],
                           config['random_start'],
                           config['loss_func'])

    saver = tf.train.Saver()

    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    # x_batch = mnist.test.images
    # y_batch = mnist.test.labels
    # x_batch_adv = x_batch.copy()

    ckpt_step = args.ckpt_step
    max_ckpt = args.max_ckpt
    batch_size = args.batch_size

    path = os.path.join(args.log_prefix, model_name + ".log")
    print(path)
    log_file = open(path, 'w')

    with tf.Session() as sess:
        for cur_ckpt in range(0, max_ckpt, ckpt_step):

            model_ckpt = os.path.join(model_dir, "checkpoint-" + str(cur_ckpt))
            print("restore: {}".format(model_ckpt))
            saver.restore(sess, model_ckpt)

            total_nat_corr = 0
            total_adv_corr = 0

            for bstart in range(0, 10000, batch_size):
                bend = min(bstart + batch_size, 10000)
                print(mnist.test.images.shape)
                x_batch = mnist.test.images[bstart:bend]
                y_batch = mnist.test.labels[bstart:bend]
                x_batch_adv = attack.perturb(x_batch, y_batch, sess)
                nat_dict = {model.x_input: x_batch,
                            model.y_input: y_batch}
                adv_dict = {model.x_input: x_batch_adv,
                            model.y_input: y_batch}

                nat_corr = sess.run(model.num_correct,
                                        feed_dict=nat_dict)
                adv_corr = sess.run(model.num_correct,
                                    feed_dict=adv_dict)

                print("batch {} nat corr: {}".format(bstart, nat_corr))
                print("batch {} adv corr: {}".format(bstart, adv_corr))

                total_nat_corr += nat_corr
                total_adv_corr += adv_corr

            nat_acc = total_nat_corr / 10000
            adv_acc = total_adv_corr / 10000

            print("natural accuracy:     {}".format(nat_acc))
            print("adversarial accuracy:     {}".format(adv_acc))

            log_file.write("{} {} {}\n".format(cur_ckpt, nat_acc, adv_acc))

        log_file.close()
