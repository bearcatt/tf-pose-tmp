import abc
import math
import os.path as osp
from collections import OrderedDict as dict

import numpy as np
import setproctitle
import tensorflow as tf

from config import cfg
from gen_batch import generate_batch
from tfflat.data_provider import (
    DataFromList, BatchData,
    MapData, MultiProcessMapDataZMQ)
from tfflat.logger import colorlogger
from tfflat.net_utils import (
    average_gradients, aggregate_batch,
    get_optimizer, get_tower_summary_dict)
from tfflat.saver import load_model, Saver
from tfflat.timer import Timer
from tfflat.utils import approx_equal


class ModelDesc(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._loss = None
        self._inputs = []
        self._outputs = []
        self._tower_summary = []

    def set_inputs(self, *vars):
        self._inputs = vars

    def set_outputs(self, *vars):
        self._outputs = vars

    def set_loss(self, var):
        if not isinstance(var, tf.Tensor):
            raise ValueError("Loss must be an single tensor.")
        self._loss = var

    def get_loss(self, include_wd=False):
        if self._loss is None:
            raise ValueError("Network doesn't define the final loss")

        if include_wd:
            weight_decay = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            weight_decay = tf.add_n(weight_decay)
            return self._loss + weight_decay
        else:
            return self._loss

    def get_inputs(self):
        if len(self._inputs) == 0:
            raise ValueError("Network doesn't define the inputs")
        return self._inputs

    def get_outputs(self):
        if len(self._outputs) == 0:
            raise ValueError("Network doesn't define the outputs")
        return self._outputs

    def add_tower_summary(self, name, vars, reduced_method='mean'):
        assert reduced_method == 'mean' or reduced_method == 'sum'
        if isinstance(vars, list):
            for v in vars:
                if vars.get_shape() == None:
                    print('Summary tensor {} got an unknown shape.'.format(name))
                else:
                    assert v.get_shape().as_list() == []
                tf.add_to_collection(name, v)
        else:
            if vars.get_shape() == None:
                print('Summary tensor {} got an unknown shape.'.format(name))
            else:
                assert vars.get_shape().as_list() == []
            tf.add_to_collection(name, vars)
        self._tower_summary.append([name, reduced_method])

    @abc.abstractmethod
    def make_network(self, is_train):
        pass


class Base(object):
    __metaclass__ = abc.ABCMeta
    """
    build graph:
      _make_graph
        make_inputs
        make_network
          add_tower_summary
      get_summary
    
    train/test
    """

    def __init__(self, net, data_iter=None, log_name='logs.txt'):
        self._input_list = []
        self._output_list = []
        self._outputs = []
        self.graph_ops = None
        self.net = net
        self.summary_dict = {}
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

        # initialize tensorflow
        tfconfig = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False)
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)

        # build_graph
        self.build_graph()

        # get data iter
        self._data_iter = data_iter

    @abc.abstractmethod
    def _make_data(self):
        return

    @abc.abstractmethod
    def _make_graph(self):
        return

    def build_graph(self):
        # all variables should be in the same graph and stored in cpu.
        with tf.device('/device:CPU:0'):
            tf.set_random_seed(2333)
            self.graph_ops = self._make_graph()
            if not isinstance(self.graph_ops, list) and not isinstance(self.graph_ops, tuple):
                self.graph_ops = [self.graph_ops]
        self.summary_dict.update(get_tower_summary_dict(self.net._tower_summary))

    def load_weights(self, model):
        if osp.exists(model + '.meta') or osp.exists(model):
            self.logger.info('Initialized model weights from {} ...'.format(model))
            load_model(self.sess, model)
            if model.split('/')[-1].startswith('snapshot_'):
                self.cur_epoch = int(model[model.find('snapshot_') + 9:model.find('.ckpt')])
                self.logger.info('Current epoch is %d.' % self.cur_epoch)
        else:
            self.logger.critical('Load nothing. There is no model in path {}.'.format(model))

    def next_feed(self):
        if self._data_iter is None:
            raise ValueError('No input data.')
        feed_dict = dict()
        for inputs in self._input_list:
            blobs = next(self._data_iter)
            for i, inp in enumerate(inputs):
                inp_shape = inp.get_shape().as_list()
                if None in inp_shape:
                    feed_dict[inp] = blobs[i]
                else:
                    feed_dict[inp] = blobs[i].reshape(*inp_shape)
        return feed_dict


class Trainer(Base):

    def __init__(self, net, data_iter=None):
        self.lr_eval = cfg.lr
        self.lr = tf.Variable(cfg.lr, trainable=False)
        self._optimizer = get_optimizer(self.lr, cfg.optimizer)
        super(Trainer, self).__init__(net, data_iter, log_name='train_logs.txt')

        # make data
        self._data_iter, self.itr_per_epoch = self._make_data()

    def _make_data(self):

        if cfg.dataset == 'COCO':
            import coco
            train_data = coco.load_train_data(cfg.datadir)
        else:
            raise NotImplementedError

        data_load_thread = DataFromList(train_data)
        if cfg.multi_thread_enable:
            data_load_thread = MultiProcessMapDataZMQ(
                data_load_thread, cfg.num_thread, generate_batch, strict=True)
        else:
            data_load_thread = MapData(data_load_thread, generate_batch)
        data_load_thread = BatchData(data_load_thread, cfg.batch_size)

        data_load_thread.reset_state()
        dataiter = data_load_thread.get_data()

        return dataiter, math.ceil(len(train_data) / cfg.batch_size / cfg.num_gpus)

    def _make_graph(self):
        self.logger.info("Generating training graph on {} GPUs ...".format(cfg.num_gpus))

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(cfg.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as name_scope:
                        # loss over single GPU
                        self.net.make_network(is_train=True)
                        if i == cfg.num_gpus - 1:
                            loss = self.net.get_loss(include_wd=True)
                        else:
                            loss = self.net.get_loss()
                        self._input_list.append(self.net.get_inputs())

                        tf.get_variable_scope().reuse_variables()

                        if i == 0:
                            if cfg.num_gpus > 1:
                                self.logger.warning("BN is calculated only on single GPU.")
                            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
                            with tf.control_dependencies(extra_update_ops):
                                grads = self._optimizer.compute_gradients(loss)
                        else:
                            grads = self._optimizer.compute_gradients(loss)
                        final_grads = []
                        with tf.variable_scope('Gradient_Mult') as scope:
                            for grad, var in grads:
                                final_grads.append((grad, var))
                        tower_grads.append(final_grads)

        if len(tower_grads) > 1:
            grads = average_gradients(tower_grads)
        else:
            grads = tower_grads[0]

        apply_gradient_op = self._optimizer.apply_gradients(grads)
        train_op = tf.group(apply_gradient_op, *extra_update_ops)

        return train_op

    def train(self):
        # saver
        self.logger.info('Initialize saver ...')
        train_saver = Saver(self.sess, tf.global_variables(), cfg.model_dump_dir)

        # initialize weights
        self.logger.info('Initialize all variables ...')
        self.sess.run(tf.variables_initializer(tf.global_variables(), name='init'))

        self.load_weights(cfg.weights)

        self.logger.info('Start training ...')
        start_itr = self.cur_epoch * self.itr_per_epoch + 1
        end_itr = self.itr_per_epoch * cfg.end_epoch + 1
        for itr in range(start_itr, end_itr):
            self.tot_timer.tic()

            self.cur_epoch = itr // self.itr_per_epoch
            itr_epoch = itr % self.itr_per_epoch
            setproctitle.setproctitle('train epoch:' + str(self.cur_epoch))

            # apply current learning policy
            cur_lr = get_lr(self.cur_epoch)
            if not approx_equal(cur_lr, self.lr_eval):
                print(self.lr_eval, cur_lr)
                self.sess.run(tf.assign(self.lr, cur_lr))

            # input data
            self.read_timer.tic()
            feed_dict = self.next_feed()
            self.read_timer.toc()

            # train one step
            self.gpu_timer.tic()
            _, self.lr_eval, *summary_res = self.sess.run(
                [self.graph_ops[0], self.lr, *self.summary_dict.values()],
                feed_dict=feed_dict)
            self.gpu_timer.toc()

            itr_summary = dict()
            for i, k in enumerate(self.summary_dict.keys()):
                itr_summary[k] = summary_res[i]

            screen = [
                'Epoch %d itr %d/%d:' % (self.cur_epoch, itr_epoch),
                'lr: %g' % (self.lr_eval),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    self.tot_timer.average_time,
                    self.gpu_timer.average_time,
                    self.read_timer.average_time
                ),
                '%.2fh/epoch' % (
                        self.tot_timer.average_time / 3600. * self.itr_per_epoch
                ),
                ' '.join(map(lambda x: '%s: %.4f' % (x[0], x[1]), itr_summary.items()))
            ]

            if itr % cfg.log_display == 0:
                self.logger.info(' '.join(screen))

            if itr % self.itr_per_epoch == 0:
                train_saver.save_model(self.cur_epoch)

            self.tot_timer.toc()


def get_lr(epoch):
    for e in cfg.lr_dec_epoch:
        if epoch < e:
            break
    if epoch < cfg.lr_dec_epoch[-1]:
        i = cfg.lr_dec_epoch.index(e)
        return cfg.lr / (cfg.lr_dec_factor ** i)
    else:
        return cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))


class Tester(Base):
    def __init__(self, net, data_iter=None):
        super(Tester, self).__init__(net, data_iter, log_name='test_logs.txt')

    def next_feed(self, batch_data):
        # fill batch
        for i in range(len(batch_data)):
            batch_size = (len(batch_data[i]) + cfg.num_gpus - 1) // cfg.num_gpus
            total_batches = batch_size * cfg.num_gpus
            left_batches = total_batches - len(batch_data[i])
            if left_batches > 0:
                batch_data[i] = np.append(
                    batch_data[i],
                    np.zeros((left_batches, *batch_data[i].shape[1:])), axis=0)

        feed_dict = dict()
        batch_size = cfg.batch_size
        for j, inputs in enumerate(self._input_list):
            for i, inp in enumerate(inputs):
                feed_dict[inp] = batch_data[i][j * batch_size: (j + 1) * batch_size]

        return feed_dict

    def _make_graph(self):
        self.logger.info("Generating testing graph on {} GPUs ...".format(cfg.num_gpus))

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(cfg.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as name_scope:
                        self.net.make_network(is_train=False)
                        self._input_list.append(self.net.get_inputs())
                        self._output_list.append(self.net.get_outputs())

                        tf.get_variable_scope().reuse_variables()

        self._outputs = aggregate_batch(self._output_list)

        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(self.sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(self.sess.graph, run_meta=run_meta, cmd='op', options=opts)

        print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))

        return self._outputs

    def predict_one(self, data):
        # TODO(reduce data in limited batch)
        setproctitle.setproctitle('test epoch:' + str(self.cur_epoch))

        self.read_timer.tic()
        feed_dict = self.next_feed(data)
        self.read_timer.toc()

        self.gpu_timer.tic()
        res = self.sess.run([*self.graph_ops], feed_dict=feed_dict)
        self.gpu_timer.toc()

        if len(data[0]) < cfg.num_gpus * cfg.batch_size:
            for i in range(len(res)):
                res[i] = res[i][:len(data[0])]

        return res
