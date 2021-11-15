from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import cv2
import time


from models.yolo import *
from data.dataset import *

flags.DEFINE_string('dataset', './data/voc2012_train.tfrecord', 'path to dataset')
flags.DEFINE_string('val_dataset', './data/voc2012_val.tfrecord', 'path to validation dataset')
flags.DEFINE_string('yaml_dir', './models/yolov5s.yaml', 'path to yaml file')
flags.DEFINE_string('classes', './data/voc2012.names', 'path to classes file')
flags.DEFINE_integer('size', 640, 'image size')
flags.DEFINE_integer('epochs', 50, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_integer('img_size', 640, 'image size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 20, 'number of classes in the model')
flags.DEFINE_boolean('multi_gpu', False, 'Use if wishing to train with more than 1 GPU.')
flags.DEFINE_float('label_smoothing', 0.02, 'label smoothing')
flags.DEFINE_integer('yolo_max_boxes', 100, 'yolo max boxes')


def transform(image, label):
    label_encoder = anchor_label.encode(label)
    return image, label_encoder
        
def main(_argv):
    train_dataset = load_tfrecord_dataset(FLAGS.batch_size,
        FLAGS.dataset, FLAGS.classes, FLAGS.size)
    Yolo = Model(cfg=FLAGS.yaml_dir)
    anchors = Yolo.model[-1].anchors
    stride = Yolo.model[-1].stride
    num_classes = FLAGS.num_classes
    anchor_label = AnchorLabeler(anchors,
                                          grids=FLAGS.img_size / stride,
                                          img_size=FLAGS.img_size,
                                          assign_method='wh',
                                          extend_offset='True')


    train_dataset = train_dataset.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    Yolo_loss = Loss(anchors, iou_thres=0.3,
                        num_classes=num_classes,
                        label_smoothing=FLAGS.label_smoothing,
                        img_size=FLAGS.img_size)
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    Yolo = Yolo(FLAGS.img_size)
    for epoch in range(0, FLAGS.epochs):
        for step, (image, target) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                output = Yolo(image)             
                iou_loss, conf_loss, prob_loss = Yolo_loss(target, output)
                pred_loss = iou_loss+conf_loss+prob_loss
                total_loss = tf.reduce_sum(pred_loss)
            grads = tape.gradient(total_loss, Yolo.trainable_variables)
            optimizer.apply_gradients(zip(grads, Yolo.trainable_variables))
            logging.info("{}_train_{}, {}, {}".format(epoch, step, total_loss.numpy(),
                list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            tf.saved_model.save(Yolo, './weights/')


if __name__=='__main__':
    app.run(main)