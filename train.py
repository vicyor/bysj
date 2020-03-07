#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:08:02 2018

@author: zwx
"""
#训练解码器
from argparse import ArgumentParser
from model import WTC_Model


parser = ArgumentParser()
#目标层
parser.add_argument('--target_layer', type=str,
                        dest='target_layer', help='target_layer(such as relu5)',
                        metavar='target_layer', required=True)
#VGG模型路径
parser.add_argument('--pretrained_path',type=str,
                        dest='pretrained_path',help='the pretrained vgg19 path',
                        metavar='Pretrained',required = True)
#训练轮数
parser.add_argument('--max_iterator',type=int,
                        dest='max_iterator',help='the max iterator',
                        metavar='MAX',required = True)
#检查点路径
parser.add_argument('--checkpoint_path',type=str,
                        dest='checkpoint_path',help='checkpoint path',
                        metavar='CheckPoint',required = True)
#训练集路径
parser.add_argument('--tfrecord_path',type=str,
                        dest='tfrecord_path',help='tfrecord path',
                        metavar='Tfrecord',required = True)
#batch_size 批处理
parser.add_argument('--batch_size',type=int,
                        dest='batch_size',help='batch_size',
                        metavar='Batch_size',required = True)
    


def main():
    #解析验证参数
    opts = parser.parse_args()
    #创建模型对象
    model = WTC_Model(target_layer = opts.target_layer,
                      pretrained_path = opts.pretrained_path,
                      max_iterator = opts.max_iterator,
                      checkpoint_path = opts.checkpoint_path,
                      tfrecord_path = opts.tfrecord_path,
                      batch_size = opts.batch_size)
    #解码器训练
    model.train()
    
if __name__=='__main__' :
    main()