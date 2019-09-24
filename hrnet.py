# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from config import cfg

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


################################################################################
# Convenience functions for building the HRNet model.
################################################################################
def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.compat.v1.layers.batch_normalization(
    inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
    scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.compat.v1.layers.conv2d(
    inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
    padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
    kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(cfg.weight_decay),
    data_format=data_format)


################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
                       data_format):
  """A single block for ResNet v1, without a bottleneck.

  Convolution then batch normalization then ReLU as described by:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=3, strides=strides,
    data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=3, strides=1,
    data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides, data_format):
  """A single block for ResNet v1, with a bottleneck.

  Similar to _building_block_v1(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=1, strides=1,
    data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=3, strides=strides,
    data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
    data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def block_layer(inputs, inp_filters, filters, bottleneck, block_fn,
                blocks, strides, training, name, data_format):
  """Creates one layer of blocks for the HRNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    inp_filters: The number of input channels for the first convolutional
      layer.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
      inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
      data_format=data_format)

  shortcut_fn = None
  if strides > 1 or filters_out != inp_filters:
    shortcut_fn = projection_shortcut

  # Only the first block per block_layer uses projection_shortcut and strides
  with tf.variable_scope('block0'):
    inputs = block_fn(inputs, filters, training, shortcut_fn, strides,
                      data_format)

  for i in range(1, blocks):
    with tf.variable_scope('block{}'.format(i)):
      inputs = block_fn(inputs, filters, training, None, 1, data_format)

  return tf.identity(inputs, name)


def HighResolutionModule(inputs, num_branches, num_inchannels,
                         num_channels, bottleneck, block_fn,
                         num_blocks, training, name, data_format,
                         multi_scale_output=True):
  """Creates one high-resolution module for the HRNet model.

  Args:
    inputs: A list of tensor of size [batch, channels, height_in, width_in]
      or [batch, height_in, width_in, channels] depending on data_format.
    num_branches: The number of branches.
    num_inchannels: A list, the number of input channels for each branch.
    num_channels: A list, the number of output channels for each branch.
      If adopting 'bottleneck_block', the output channels will be 4 times.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    num_blocks: The number of blocks contained in each branch.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
    multi_scale_output: Output all the branches or only the high-resolution
      branch.

  Returns:
    If multi_scale_output, the output is a list including the output tensors
    of all the branches. Else, the output is a list including the output
    tensors of the high-resolution branch.
  """
  if bottleneck:
    num_outchannels = [c * 4 for c in num_channels]
  else:
    num_outchannels = num_channels

  strides = 1
  # compute the output for each branch
  outputs = []
  for i in range(num_branches):
    branch_name = name + '_branch_{}'.format(i)
    with tf.variable_scope(branch_name):
      branch = block_layer(
        inputs=inputs[i], inp_filters=num_inchannels[i],
        filters=num_channels[i], bottleneck=bottleneck,
        block_fn=block_fn, blocks=num_blocks,
        strides=strides, training=training,
        name=branch_name, data_format=data_format)

    outputs.append(branch)

  if num_branches == 1:
    return outputs

  def fusion(inputs, inp_index, out_index):
    """Create multi-resolution fusion module

    Args:
      inputs: A list of tensor of size [batch, channels, height_in, width_in]
        or [batch, height_in, width_in, channels] depending on data_format.
      inp_index: The indexes of inputs that are fused together.
      out_index: The index of the output, which decides the output resolution.

    Return:
      The fused output.
    """
    shortcut = inputs[out_index]
    if data_format == 'channel_first':
      _, _, out_h, out_w = shortcut.get_shape().as_list()
    else:
      _, out_h, out_w, _ = shortcut.get_shape().as_list()

    for ind in inp_index:
      input_branch = inputs[ind]
      with tf.variable_scope('input_{}_output_{}'.format(ind, out_index)):
        if ind > out_index:
          input_branch = conv2d_fixed_padding(
            inputs=input_branch, filters=num_outchannels[out_index],
            kernel_size=1, strides=1, data_format=data_format)
          input_branch = batch_norm(input_branch, training, data_format)
          input_branch = tf.nn.relu(input_branch)
          if data_format == 'channel_first':
            input_branch = tf.transpose(input_branch, perm=[0, 2, 3, 1])
            input_branch = tf.image.resize(input_branch, (out_h, out_w),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            input_branch = tf.transpose(input_branch, perm=[0, 3, 1, 2])
          else:
            input_branch = tf.image.resize(input_branch, (out_h, out_w),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        elif ind == out_index:
          continue
        else:
          for k in range(out_index - ind):
            if k == out_index - ind - 1:
              input_branch = conv2d_fixed_padding(
                inputs=input_branch, filters=num_outchannels[out_index],
                kernel_size=3, strides=2, data_format=data_format)
              input_branch = batch_norm(input_branch, training, data_format)
            else:
              input_branch = conv2d_fixed_padding(
                inputs=input_branch, filters=num_outchannels[out_index],
                kernel_size=3, strides=2, data_format=data_format)
              input_branch = batch_norm(input_branch, training, data_format)
              input_branch = tf.nn.relu(input_branch)

      shortcut += input_branch
    shortcut = tf.nn.relu(shortcut)
    return shortcut

  with tf.variable_scope('ms_fusion_' + name):
    if not multi_scale_output:
      fused_outputs = [fusion(outputs, list(range(num_branches)), 0)]
    else:
      fused_outputs = []
      for i in range(num_branches):
        fused_outputs.append(fusion(outputs, list(range(num_branches)), i))
  return fused_outputs


def Transition(inputs, num_outchannels, training, data_format):
  """Create a new branch with 2x depth and 1/2 resolution.
  """
  trans_output = conv2d_fixed_padding(
    inputs=inputs, filters=num_outchannels,
    kernel_size=3, strides=2, data_format=data_format)
  trans_output = batch_norm(trans_output, training, data_format)
  trans_output = tf.nn.relu(trans_output)
  return trans_output


class Model(object):
  """Base class for building the HRNet Model."""

  def __init__(self, hrnet_size, bottleneck, num_filters,
               kernel_size, conv_stride, module_sizes, block_sizes,
               data_format=None, dtype=DEFAULT_DTYPE):
    """Creates a model for classifying an image.

    Args:
      hrnet_size: A single integer for the size of the HRNet model.
      bottleneck: Use regular blocks or bottleneck blocks.
      num_filters: The number of filters to use for the first block layer
        of the model. This number is then doubled for each subsequent block
        layer.
      kernel_size: The kernel size to use for stem convolution.
      conv_stride: stride size for the initial convolutional layer\
      module_sizes: A list containing n values, where n is the number of sets of
        modules desired. Each value should be the number of modules in the
        i-th set.
      block_sizes: A list containing n values, where n is the number of sets of
        block layers desired. Each value should be the number of blocks in the
        i-th set.
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
      dtype: The TensorFlow dtype to use for calculations. If not specified
        tf.float32 is used.

    Raises:
      ValueError: if invalid version is selected.
    """
    self.hrnet_size = hrnet_size

    if not data_format:
      data_format = 'channels_last'

    self.bottleneck = bottleneck
    if bottleneck:
      self.block_fn = _bottleneck_block_v1
    else:
      self.block_fn = _building_block_v1

    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    self.data_format = data_format
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.module_sizes = module_sizes
    self.block_sizes = block_sizes
    self.dtype = dtype

  def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):
    """Creates variables in fp32, then casts to fp16 if necessary.

    This function is a custom getter. A custom getter is a function with the
    same signature as tf.get_variable, except it has an additional getter
    parameter. Custom getters can be passed as the `custom_getter` parameter of
    tf.variable_scope. Then, tf.get_variable will call the custom getter,
    instead of directly getting a variable itself. This can be used to change
    the types of variables that are retrieved with tf.get_variable.
    The `getter` parameter is the underlying variable getter, that would have
    been called if no custom getter was used. Custom getters typically get a
    variable with `getter`, then modify it in some way.

    This custom getter will create an fp32 variable. If a low precision
    (e.g. float16) variable was requested it will then cast the variable to the
    requested dtype. The reason we do not directly create variables in low
    precision dtypes is that applying small gradients to such variables may
    cause the variable not to change.

    Args:
      getter: The underlying variable getter, that has the same signature as
        tf.get_variable and returns a variable.
      name: The name of the variable to get.
      shape: The shape of the variable to get.
      dtype: The dtype of the variable to get. Note that if this is a low
        precision dtype, the variable will be created as a tf.float32 variable,
        then cast to the appropriate dtype
      *args: Additional arguments to pass unmodified to getter.
      **kwargs: Additional keyword arguments to pass unmodified to getter.

    Returns:
      A variable which is cast to fp16 if necessary.
    """

    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)

  def _model_variable_scope(self):
    """Returns a variable scope that the model should be created under.

    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.

    Returns:
      A variable scope for the model.
    """

    return tf.compat.v1.variable_scope('hrnet_model',
                                       custom_getter=self._custom_dtype_getter)

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """

    with self._model_variable_scope():
      if self.data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])

      with tf.variable_scope('stem'):
        inputs = conv2d_fixed_padding(
          inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
          strides=self.conv_stride, data_format=self.data_format)
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)
        inputs = tf.identity(inputs, 'stem_conv1')

        inputs = conv2d_fixed_padding(
          inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
          strides=self.conv_stride, data_format=self.data_format)
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)
        inputs = tf.identity(inputs, 'stem_conv2')

      # stage 1
      with tf.variable_scope('stage1'):
        stage1_out = block_layer(
          inputs=inputs, inp_filters=self.num_filters,
          filters=self.num_filters, bottleneck=True,
          block_fn=_bottleneck_block_v1, blocks=self.block_sizes[0],
          strides=1, training=training,
          name='stage1', data_format=self.data_format)

      pre_stage_channels = self.num_filters * 4
      stagex_input = [stage1_out]
      num_inchannels = [pre_stage_channels]

      # stage 2 to last stage
      for i, num_modules in enumerate(self.module_sizes[1:], 2):
        with tf.variable_scope('stage{}'.format(i)):
          num_channels = [self.hrnet_size * 2 ** n for n in range(i)]
          transn = Transition(
            stagex_input[-1], num_outchannels=num_channels[-1],
            training=training, data_format=self.data_format)
          stagex_input.append(transn)
          num_inchannels.append(num_inchannels[-1])

          for j in range(num_modules):
            if i == len(self.module_sizes) and j == num_modules - 1:
              multi_scale_output = False
            else:
              multi_scale_output = True

            stagex_input = HighResolutionModule(
              stagex_input, num_branches=i,
              num_inchannels=num_inchannels, num_channels=num_channels,
              bottleneck=self.bottleneck, block_fn=self.block_fn,
              num_blocks=self.block_sizes[i - 1], training=training,
              name='stage{}_block{}'.format(i, j), data_format=self.data_format,
              multi_scale_output=multi_scale_output)
          num_inchannels = num_channels

      return stagex_input


def HRNet(hrnet_size, input, training):
  model = Model(hrnet_size,
                bottleneck=False,
                num_filters=64,
                kernel_size=3,
                conv_stride=2,
                module_sizes=[1, 1, 4, 3],
                block_sizes=[4, 4, 4, 4])
  out = model(input, training)
  return out


if __name__ == "__main__":
  import numpy as np

  input_ = np.random.rand(4, 224, 224, 3)
  input = tf.placeholder(tf.float32, [4, 224, 224, 3], 'images')
  out = HRNet(32, input, True)
  with tf.Session() as sess:
    writer = tf.summary.FileWriter("output", sess.graph)
    sess.run(tf.global_variables_initializer())
    print(sess.run(out, feed_dict={input: input_}))
    writer.close()
