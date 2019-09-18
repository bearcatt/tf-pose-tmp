import tensorflow as tf

from config import cfg
from hrnet import HRNet
from .engine import ModelDesc


class Model(ModelDesc):

    def head_net(self, features):

        out = tf.compat.v1.layers.conv2d(
            inputs=features[0], filters=cfg.num_kps, kernel_size=[1, 1], 
            padding='SAME', data_format='channels_last', use_bias=True,
            kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
            bias_regularizer=None, trainable=True, name='out')

        return out

    def render_gaussian_heatmap(self, coord, output_shape, sigma):

        x = [i for i in range(output_shape[1])]
        y = [i for i in range(output_shape[0])]
        xx, yy = tf.meshgrid(x, y)
        xx = tf.reshape(tf.to_float(xx), (1, *output_shape, 1))
        yy = tf.reshape(tf.to_float(yy), (1, *output_shape, 1))

        x = tf.floor(tf.reshape(coord[:, :, 0], [-1, 1, 1, cfg.num_kps]) \
                / cfg.input_shape[1] * output_shape[1] + 0.5)
        y = tf.floor(tf.reshape(coord[:, :, 1], [-1, 1, 1, cfg.num_kps]) \
                / cfg.input_shape[0] * output_shape[0] + 0.5)

        heatmap = tf.exp(-(((xx - x) / tf.to_float(sigma)) ** 2) / tf.to_float(2) - (
                ((yy - y) / tf.to_float(sigma)) ** 2) / tf.to_float(2))

        return heatmap * 255.

    def make_network(self, is_train):

        if is_train:
            image = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.input_shape, 3])
            target_coord = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_kps, 2])
            valid = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_kps])
            self.set_inputs(image, target_coord, valid)
        else:
            image = tf.placeholder(tf.float32, shape=[None, *cfg.input_shape, 3])
            self.set_inputs(image)

        hrnet_fms = HRNet(cfg.hrnet_size, image, is_train)
        heatmap_outs = self.head_net(hrnet_fms)

        if is_train:
            gt_heatmap = tf.stop_gradient(
                self.render_gaussian_heatmap(target_coord, cfg.output_shape, cfg.sigma))
            valid_mask = tf.reshape(valid, [cfg.batch_size, 1, 1, cfg.num_kps])
            loss = tf.reduce_mean(tf.square(heatmap_outs - gt_heatmap) * valid_mask)
            self.add_tower_summary('loss', loss)
            self.set_loss(loss)
        else:
            self.set_outputs(heatmap_outs)
