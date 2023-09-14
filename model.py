import numpy as np
import scipy.io as sio
import tensorflow as tf

def relu(input):
    relu = tf.nn.relu(input)
    # convert nan to zero (nan != nan)
    nan_to_zero = tf.where(tf.equal(relu, relu), relu, tf.zeros_like(relu))
    return nan_to_zero

def conv3d(x, input_filters, output_filters, kernel, strides,name):
    with tf.variable_scope(name):
        shape = [kernel,kernel, kernel, input_filters, output_filters]
        weight = tf.get_variable('w', shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),regularizer=None, trainable=True, collections=None)
        return tf.nn.conv3d(x, weight, strides=[1, strides,strides, strides, 1], padding='SAME',name='conv')
def conv3d_b(x, input_filters, output_filters, kernel, strides,name ):
    with tf.variable_scope(name):
        shape = [kernel, kernel, kernel, input_filters, output_filters]
        bias_shape = [output_filters]
        weight = tf.get_variable('w', shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),regularizer=None, trainable=True, collections=None)
        biases = tf.get_variable('b', shape=bias_shape, dtype=tf.float32, initializer=tf.zeros_initializer(),regularizer=None, trainable=True, collections=None)
        return tf.nn.conv3d(x, weight, strides=[1, strides, strides, strides, 1], padding='SAME',
                            name='conv')+biases
def deconv3d(x, input_filters, output_filters, kernel, strides,name):
    with tf.variable_scope(name):
        shape = [kernel, kernel,kernel, output_filters,input_filters]
        weight = tf.get_variable('w', shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), regularizer=None, trainable=True, collections=None)
        batch_size = tf.shape(x)[0]
        depth=tf.shape(x)[1] * strides
        height = tf.shape(x)[2] * strides
        width = tf.shape(x)[3] * strides
        outshape=[batch_size,depth,height,width,output_filters]
        return tf.nn.conv3d_transpose(x, weight, output_shape=outshape,strides=[1, strides,strides, strides, 1], padding='SAME',name='deconv')
def deconv3d_b(x, input_filters, output_filters, kernel, strides,name):
    with tf.variable_scope(name):
        shape = [kernel, kernel,kernel, output_filters,input_filters]
        bias_shape = [output_filters]
        weight = tf.get_variable('w', shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), regularizer=None, trainable=True, collections=None)
        biases = tf.get_variable('b', shape=bias_shape, dtype=tf.float32, initializer=tf.zeros_initializer(), regularizer=None, trainable=True, collections=None)
        batch_size = tf.shape(x)[0]
        depth=tf.shape(x)[1] * strides
        height = tf.shape(x)[2] * strides
        width = tf.shape(x)[3] * strides
        outshape=[batch_size,depth,height,width,output_filters]
        return tf.nn.conv3d_transpose(x, weight, output_shape=outshape,strides=[1, strides,strides, strides, 1], padding='SAME',
                            name='deconv')+biases

def max_pool3d(input):
  return tf.nn.max_pool3d(input, ksize=[1,2, 2, 2, 1], strides=[1,2, 2, 2, 1], padding='SAME')

def avg_pool3d(input):
  return tf.nn.avg_pool3d(input, ksize=[1,2, 2, 2, 1], strides=[1,2, 2, 2, 1], padding='SAME')

def SA_norm(last,input,out_c,name):
    field=tf.expand_dims(input[:,:,:,:,0],4)
    mag = tf.expand_dims(input[:, :, :, :, 1], 4)

    f1=relu(conv3d_b(field,1,32,3,1,'f1'+name))
    m1 = relu(conv3d_b(mag, 1, 32, 3, 1, 'm1' + name))

    f_concat = tf.concat([f1,m1], 4)
    f_weight = tf.get_variable('fw'+name, shape=[64], dtype=tf.float32, initializer=tf.ones_initializer(),
                                regularizer=None, trainable=True, collections=None)

    fin = relu(
        conv3d_b(f_concat * f_weight, 64, 32, 1, 1, 'fin'+name))

    g=conv3d_b(fin,32,out_c,3,1,'g'+name)
    b = conv3d_b(fin, 32, out_c, 3, 1, 'b'+name)
    out=last*(1+g)+b
    return out

def SA_norm_NoWeight(last,input,out_c,name):

    field=tf.expand_dims(input[:,:,:,:,0],4)
    mag= tf.expand_dims(input[:, :, :, :, 1], 4)

    f1=relu(conv3d_b(field,1,32,3,1,'f1'+name))
    m1 = relu(conv3d_b( mag, 1, 32, 3, 1, 'm1' + name))

    concat = tf.concat([f1,m1], 4)

    fin = relu(
        conv3d_b(concat, 64, 32, 1, 1, 'fin'+name))

    shared=fin
    g=conv3d_b(shared,32,out_c,3,1,'g'+name)
    b = conv3d_b(shared, 32, out_c, 3, 1, 'b'+name)
    out=last*(1+g)+b
    return out


def mse_gdl_loss(cosout,coslabel):
    loss1=tf.reduce_mean(tf.square(cosout-coslabel))
    loss3=0.00000001*gdl3d(cosout,coslabel)
    loss=loss1+loss3
    return loss


def gdl3d(gen_frames, gt_frames, alpha=1):
    """
    Calculates the sum of GDL losses between the predicted and ground truth frames.
    This is the 3d version.
    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param alpha: The power to which each gradient term is raised.
    @return: The GDL loss for 3d.
    """
    # calculate the loss for each scale
    scale_losses = []

    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    pos = tf.constant(np.identity(1), dtype=tf.float32)
    neg = -1 * pos

    baseFilter = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]# 2x1x1x1
    filter_x = tf.expand_dims(baseFilter, 1)  # [-1, 1] # 2x1x1x1x1
    filter_y = tf.expand_dims(baseFilter, 0)  # [-1, 1] # 1x2x1x1x1
    filter_z = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1] # 1x2x1x1
    filter_z = tf.expand_dims(filter_z, 0) # [-1, 1] #1x1x2x1x1
    strides = [1, 1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.nn.conv3d(gen_frames, filter_x, strides, padding=padding)
    gen_dy = tf.nn.conv3d(gen_frames, filter_y, strides, padding=padding)
    gen_dz = tf.nn.conv3d(gen_frames, filter_z, strides, padding=padding)
    gt_dx = tf.nn.conv3d(gt_frames, filter_x, strides, padding=padding)
    gt_dy = tf.nn.conv3d(gt_frames, filter_y, strides, padding=padding)
    gt_dz = tf.nn.conv3d(gt_frames, filter_z, strides, padding=padding)

    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y = tf.abs(gt_dy - gen_dy)
    grad_diff_z = tf.abs(gt_dz - gen_dz)

    scale_losses.append(tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha + grad_diff_z ** alpha)))
    return tf.reduce_mean(tf.stack(scale_losses))

def load_magimage(path):
    image = sio.loadmat(path)
    image = image['magmap']
    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    image = image[:, :,:, :, np.newaxis]
    image = image.astype('float32')
    return image

def load_fieldimage(path):
    image = sio.loadmat(path)
    image = image['field']
    #image=(image+1)/2
    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    image = image[:, :,:, :, np.newaxis]
    # Input to the VGG net expects the mean to be subtracted.
    image = image.astype('float32')

    return image

def load_cosmos(path):
    image = sio.loadmat(path)
    image = image['chi_mo']
    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    image = image[:, :, :,:, np.newaxis]
    image = image.astype('float32')

    return image

def Unet_NoBN(fieldinput):
    with tf.variable_scope('syn'):
        encode1_1=relu(conv3d_b(fieldinput,1,32,3,1,'encode1_1'))
        encode1_2 = relu(conv3d_b(encode1_1, 32, 32, 3, 1, 'encode1_2'))

        encode2_1=relu(conv3d_b(encode1_2, 32, 32, 3, 2, 'encode2_1'))
        encode2_2 = relu(conv3d_b(encode2_1, 32, 64, 3, 1, 'encode2_2'))
        encode2_3 = relu(conv3d_b(encode2_2, 64, 64, 3, 1, 'encode2_3'))

        encode3_1=relu(conv3d_b(encode2_3, 64, 64, 3, 2, 'encode3_1'))
        encode3_2 = relu(conv3d_b(encode3_1, 64, 128, 3, 1, 'encode3_2'))
        encode3_3 = relu(conv3d_b(encode3_2, 128, 128, 3, 1, 'encode3_3'))

        encode4_1 = relu(conv3d_b(encode3_3, 128, 128, 3, 2, 'encode4_1'))
        encode4_2 = relu(conv3d_b(encode4_1, 128, 256, 3, 1, 'encode4_2'))
        encode4_3 = relu(conv3d_b(encode4_2, 256, 256, 3, 1, 'encode4_3'))
        encode4_4 = relu(conv3d_b(encode4_3, 256, 256, 3, 1, 'encode4_4'))

        decode3_1 = relu(deconv3d_b(encode4_4, 256, 128, 3, 2, 'decode3_1'))
        decode3_2 = tf.concat([decode3_1, encode3_3], 4)
        decode3_4 = relu(conv3d_b(decode3_2, 256, 128, 3, 1, 'decode3_4'))

        decode3_5 = relu(
            conv3d_b(decode3_4, 128, 128, 3, 1, 'decode3_5') )

        decode4_1 = relu(
            deconv3d_b(decode3_5, 128, 64, 3, 2, 'decode4_1'))
        decode4_2 = tf.concat([decode4_1, encode2_3], 4)
        decode4_3 = relu(conv3d_b(decode4_2, 128, 64, 3, 1, 'decode4_3'))

        decode4_4 = relu(
            conv3d_b(decode4_3, 64, 64, 3, 1, 'decode4_4'))

        decode5_1 = relu(deconv3d_b(decode4_4, 64, 32, 3, 2, 'decode5_1'))
        decode5_2 = tf.concat([decode5_1, encode1_2], 4)
        decode5_3 = relu(conv3d_b(decode5_2, 64, 32, 3, 1, 'decode5_3'))

        decode5_4 = relu(conv3d_b(decode5_3, 32, 32, 3, 1, 'decode5_4'))
        cosout = conv3d_b(decode5_4, 32, 1, 1, 1, 'out')

        return cosout

def SAQSM(mag_input,field_input,is_train):
    with tf.variable_scope('syn'):
        field_conv1 = relu(
            tf.layers.batch_normalization(conv3d(field_input, 1, 48, 3, 1, 'field_conv1'), training=is_train,
                                          name='bn_field1'))
        field_conv2_33 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 3, 1, 'field_conv2_33'), training=is_train,
                                          name='bn_field2_33'))
        field_conv2_11 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 1, 1, 'field_conv2_11'), training=is_train,
                                          name='bn_field2_11'))
        field_conv2_55 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 3, 1, 'field_conv2_55'), training=is_train,
                                          name='bn_field2_55'))
        field_conv2_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv2_55, 16, 16, 3, 1, 'field_conv2_55_1'), training=is_train,
                                          name='bn_field2_55_1'))
        field_cat1 = tf.concat([field_conv2_33, field_conv2_11, field_conv2_55_1], 4)

        field_conv3_33 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 3, 1, 'field_conv3_33'), training=is_train,
                                          name='bn_field3_33'))
        field_conv3_11 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 1, 1, 'field_conv3_11'), training=is_train,
                                          name='bn_field3_11'))
        field_conv3_55 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 3, 1, 'field_conv3_55'), training=is_train,
                                          name='bn_field3_55'))
        field_conv3_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv3_55, 16, 16, 3, 1, 'field_conv3_55_1'), training=is_train,
                                          name='bn_field3_55_1'))
        field_cat2 = tf.concat([field_conv3_33, field_conv3_11, field_conv3_55_1], 4)

        field_conv4_33 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 3, 1, 'field_conv4_33'), training=is_train,
                                          name='bn_field4_33'))
        field_conv4_11 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 1, 1, 'field_conv4_11'), training=is_train,
                                          name='bn_field4_11'))
        field_conv4_55 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 3, 1, 'field_conv4_55'), training=is_train,
                                          name='bn_field4_55'))
        field_conv4_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv4_55, 16, 16, 3, 1, 'field_conv4_55_1'), training=is_train,
                                          name='bn_field4_55_1'))
        field_cat3 = tf.concat([field_conv4_33, field_conv4_11, field_conv4_55_1], 4)


        field_concat = tf.concat([field_cat1, field_cat2, field_cat3], 4)
        field_weight = tf.get_variable('field_w', shape=[144], dtype=tf.float32, initializer=tf.ones_initializer(),
                                      regularizer=None, trainable=True, collections=None)
        
        field_weight_input = relu(tf.layers.batch_normalization(conv3d(field_concat * field_weight, 144, 32, 1, 1, 'field_weight_input'), training=is_train,
                                                     name='bn_field_weight_input'))

        encode1_1 = relu(tf.layers.batch_normalization(conv3d(field_weight_input, 32, 32, 3, 1, 'encode1_1'), training=is_train,
                                                       name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=is_train,
                                                       name='bnen1_2'))

        encode2_1 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 2, 'encode2_1'), training=is_train,
                                                       name='bnencode2_1'))
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=is_train,
                                                       name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=is_train,
                                                       name='bnen2_3'))

        encode3_1 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 2, 'encode3_1'), training=is_train,
                                                       name='bnencode3_1'))
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=is_train,
                                                       name='bnen3_2'))
        encode3_3 = relu(
            tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=is_train,
                                          name='bnen3_3'))

        encode4_1 = relu(
            tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 2, 'encode4_1'), training=is_train,
                                          name='bnencode4_1'))
        encode4_2 = relu(
            tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_2'), training=is_train,
                                          name='bnen4_2'))
        
        SAM_input1 = tf.concat([field_input, mag_input], 4)
        SAM_input2 = avg_pool3d(SAM_input1)
        SAM_input3 = avg_pool3d(SAM_input2)
        SAM_input4 = avg_pool3d(SAM_input3)
        
        encode4_3 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(encode4_2, 256, 256, 3, 1, 'encode4_3'), training=is_train,
                                          name='bnen4_3'), SAM_input4, 256, 'adencode4_3'))
        encode4_4 = relu(
            tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=is_train,
                                          name='bnen4_4') + encode4_3)

        decode3_1 = relu(deconv3d(encode4_4, 256, 128, 3, 2, 'decode3_1'))
        decode3_2 = tf.concat([decode3_1, encode3_3], 4)

        decode3_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=is_train,
                                          name='bnde3_4'), SAM_input3, 128, 'addecode3_4'))
        decode3_6 = relu(
            tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_6'),
                                          training=is_train, name='bnde3_6') + decode3_4)

        decode4_1 = relu(
            deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'))
        decode4_2 = tf.concat([decode4_1, encode2_3], 4)
        decode4_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=is_train,
                                          name='bnde4_4'), SAM_input2, 64, 'addecode4_4'))
        decode4_6 = relu(
            tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_6'), training=is_train,
                                          name='bnde4_6') + decode4_4)

        decode5_1 = relu(
            deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'))
        decode5_2 = tf.concat([decode5_1, encode1_2], 4)
        decode5_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=is_train,
                                          name='bnde5_4'), SAM_input1, 32, 'addecode5_4'))
        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_6'), training=is_train,
                                                       name='bnde5_6') + decode5_4)

        cosout = conv3d_b(decode5_6, 32, 1, 1, 1, 'out')
        return cosout

def SAQSM_3SAM(mag_input,field_input,is_train):
    with tf.variable_scope('syn'):
        field_conv1 = relu(
            tf.layers.batch_normalization(conv3d(field_input, 1, 48, 3, 1, 'field_conv1'), training=is_train,
                                          name='bn_field1'))
        field_conv2_33 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 3, 1, 'field_conv2_33'), training=is_train,
                                          name='bn_field2_33'))
        field_conv2_11 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 1, 1, 'field_conv2_11'), training=is_train,
                                          name='bn_field2_11'))
        field_conv2_55 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 3, 1, 'field_conv2_55'), training=is_train,
                                          name='bn_field2_55'))
        field_conv2_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv2_55, 16, 16, 3, 1, 'field_conv2_55_1'), training=is_train,
                                          name='bn_field2_55_1'))
        field_cat1 = tf.concat([field_conv2_33, field_conv2_11, field_conv2_55_1], 4)

        field_conv3_33 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 3, 1, 'field_conv3_33'), training=is_train,
                                          name='bn_field3_33'))
        field_conv3_11 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 1, 1, 'field_conv3_11'), training=is_train,
                                          name='bn_field3_11'))
        field_conv3_55 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 3, 1, 'field_conv3_55'), training=is_train,
                                          name='bn_field3_55'))
        field_conv3_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv3_55, 16, 16, 3, 1, 'field_conv3_55_1'), training=is_train,
                                          name='bn_field3_55_1'))
        field_cat2 = tf.concat([field_conv3_33, field_conv3_11, field_conv3_55_1], 4)

        field_conv4_33 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 3, 1, 'field_conv4_33'), training=is_train,
                                          name='bn_field4_33'))
        field_conv4_11 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 1, 1, 'field_conv4_11'), training=is_train,
                                          name='bn_field4_11'))
        field_conv4_55 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 3, 1, 'field_conv4_55'), training=is_train,
                                          name='bn_field4_55'))
        field_conv4_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv4_55, 16, 16, 3, 1, 'field_conv4_55_1'), training=is_train,
                                          name='bn_field4_55_1'))
        field_cat3 = tf.concat([field_conv4_33, field_conv4_11, field_conv4_55_1], 4)

        field_concat = tf.concat([field_cat1, field_cat2, field_cat3], 4)
        field_weight = tf.get_variable('field_w', shape=[144], dtype=tf.float32, initializer=tf.ones_initializer(),
                                       regularizer=None, trainable=True, collections=None)

        field_weight_input = relu(
            tf.layers.batch_normalization(conv3d(field_concat * field_weight, 144, 32, 1, 1, 'field_weight_input'),
                                          training=is_train,
                                          name='bn_field_weight_input'))

        encode1_1 = relu(
            tf.layers.batch_normalization(conv3d(field_weight_input, 32, 32, 3, 1, 'encode1_1'), training=is_train,
                                          name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=is_train,
                                                       name='bnen1_2'))

        encode2_1 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 2, 'encode2_1'), training=is_train,
                                                       name='bnencode2_1'))
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=is_train,
                                                       name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=is_train,
                                                       name='bnen2_3'))

        encode3_1 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 2, 'encode3_1'), training=is_train,
                                                       name='bnencode3_1'))
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=is_train,
                                                       name='bnen3_2'))
        encode3_3 = relu(
            tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=is_train,
                                          name='bnen3_3'))

        encode4_1 = relu(
            tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 2, 'encode4_1'), training=is_train,
                                          name='bnencode4_1'))
        encode4_2 = relu(
            tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_2'), training=is_train,
                                          name='bnen4_2'))

        encode4_3 = relu(tf.layers.batch_normalization(conv3d(encode4_2, 256, 256, 3, 1, 'encode4_3'), training=is_train, name='bnen4_3'))
        encode4_4 = relu(
            tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=is_train,
                                          name='bnen4_4'))
        SAM_input1 = tf.concat([field_input, mag_input], 4)
        SAM_input2 = avg_pool3d(SAM_input1)
        SAM_input3 = avg_pool3d(SAM_input2)

        decode3_1 = relu(deconv3d(encode4_4, 256, 128, 3, 2, 'decode3_1'))
        decode3_2 = tf.concat([decode3_1, encode3_3], 4)
        decode3_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=is_train,
                                          name='bnde3_4'), SAM_input3, 128, 'addecode3_4'))
        decode3_6 = relu(
            tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_6') ,
                                          training=is_train, name='bnde3_6')+decode3_4)

        decode4_1 = relu(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'))
        decode4_2 = tf.concat([decode4_1, encode2_3], 4)
        decode4_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=is_train,
                                          name='bnde4_4'), SAM_input2, 64, 'addecode4_4'))

        decode4_6 = relu(
            tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_6') , training=is_train,
                                          name='bnde4_6')+ decode4_4)

        decode5_1 = relu(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'))
        decode5_2 = tf.concat([decode5_1, encode1_2], 4)
        decode5_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=is_train,
                                          name='bnde5_4'), SAM_input1, 32, 'addecode5_4'))

        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_6'), training=is_train,
                                          name='bnde5_6')+ decode5_4)
        cosout = conv3d_b(decode5_6, 32, 1, 1, 1, 'out')

        return cosout
def SAQSM_2SAM(mag_input,field_input,is_train):
    with tf.variable_scope('syn'):
        field_conv1 = relu(
            tf.layers.batch_normalization(conv3d(field_input, 1, 48, 3, 1, 'field_conv1'), training=is_train,
                                          name='bn_field1'))
        field_conv2_33 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 3, 1, 'field_conv2_33'), training=is_train,
                                          name='bn_field2_33'))
        field_conv2_11 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 1, 1, 'field_conv2_11'), training=is_train,
                                          name='bn_field2_11'))
        field_conv2_55 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 3, 1, 'field_conv2_55'), training=is_train,
                                          name='bn_field2_55'))
        field_conv2_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv2_55, 16, 16, 3, 1, 'field_conv2_55_1'), training=is_train,
                                          name='bn_field2_55_1'))
        field_cat1 = tf.concat([field_conv2_33, field_conv2_11, field_conv2_55_1], 4)

        field_conv3_33 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 3, 1, 'field_conv3_33'), training=is_train,
                                          name='bn_field3_33'))
        field_conv3_11 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 1, 1, 'field_conv3_11'), training=is_train,
                                          name='bn_field3_11'))
        field_conv3_55 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 3, 1, 'field_conv3_55'), training=is_train,
                                          name='bn_field3_55'))
        field_conv3_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv3_55, 16, 16, 3, 1, 'field_conv3_55_1'), training=is_train,
                                          name='bn_field3_55_1'))
        field_cat2 = tf.concat([field_conv3_33, field_conv3_11, field_conv3_55_1], 4)

        field_conv4_33 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 3, 1, 'field_conv4_33'), training=is_train,
                                          name='bn_field4_33'))
        field_conv4_11 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 1, 1, 'field_conv4_11'), training=is_train,
                                          name='bn_field4_11'))
        field_conv4_55 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 3, 1, 'field_conv4_55'), training=is_train,
                                          name='bn_field4_55'))
        field_conv4_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv4_55, 16, 16, 3, 1, 'field_conv4_55_1'), training=is_train,
                                          name='bn_field4_55_1'))
        field_cat3 = tf.concat([field_conv4_33, field_conv4_11, field_conv4_55_1], 4)

        field_concat = tf.concat([field_cat1, field_cat2, field_cat3], 4)
        field_weight = tf.get_variable('field_w', shape=[144], dtype=tf.float32, initializer=tf.ones_initializer(),
                                       regularizer=None, trainable=True, collections=None)

        field_weight_input = relu(
            tf.layers.batch_normalization(conv3d(field_concat * field_weight, 144, 32, 1, 1, 'field_weight_input'),training=is_train,
                                          name='bn_field_weight_input'))

        encode1_1 = relu(
            tf.layers.batch_normalization(conv3d(field_weight_input, 32, 32, 3, 1, 'encode1_1'), training=is_train,
                                          name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=is_train,
                                                       name='bnen1_2'))

        encode2_1 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 2, 'encode2_1'), training=is_train,
                                                       name='bnencode2_1'))
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=is_train,
                                                       name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=is_train,
                                                       name='bnen2_3'))

        encode3_1 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 2, 'encode3_1'), training=is_train,
                                                       name='bnencode3_1'))
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=is_train,
                                                       name='bnen3_2'))
        encode3_3 = relu(
            tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=is_train,
                                          name='bnen3_3'))

        encode4_1 = relu(
            tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 2, 'encode4_1'), training=is_train,
                                          name='bnencode4_1'))
        encode4_2 = relu(
            tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_2'), training=is_train,
                                          name='bnen4_2'))

        encode4_3 = relu(
            tf.layers.batch_normalization(conv3d(encode4_2, 256, 256, 3, 1, 'encode4_3'), training=is_train,
                                          name='bnen4_3'))
        encode4_4 = relu(
            tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=is_train,
                                          name='bnen4_4'))
        SAM_input1 = tf.concat([field_input, mag_input], 4)
        SAM_input2 = avg_pool3d(SAM_input1)

        decode3_1 = relu(deconv3d(encode4_4, 256, 128, 3, 2, 'decode3_1'),)
        decode3_2 = tf.concat([decode3_1, encode3_3], 4)
        decode3_4 = relu(
            tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=is_train,
                                          name='bnde3_4'))
        decode3_5 = relu(
            tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_5') ,
                                          training=is_train, name='bnde3_5'))

        decode4_1 = relu(deconv3d(decode3_5, 128, 64, 3, 2, 'decode2_1'))
        decode4_2 = tf.concat([decode4_1, encode2_3], 4)
        decode4_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=is_train,
                                          name='bnde4_4'), SAM_input2, 64, 'addecode4_4'))

        decode4_6 = relu(
            tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_6') , training=is_train,
                                          name='bnde4_6')+ decode4_4)

        decode5_1 = relu(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'))
        decode5_2 = tf.concat([decode5_1, encode1_2], 4)
        decode5_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=is_train,
                                          name='bnde5_4'), SAM_input1, 32, 'addecode5_4'))

        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_6'), training=is_train,
                                          name='bnde5_6')+ decode5_4)
        cosout = conv3d_b(decode5_6, 32, 1, 1, 1, 'out')
        return cosout
def SAQSM_1SAM(mag_input,field_input,is_train):
    with tf.variable_scope('syn'):
        field_conv1 = relu(
            tf.layers.batch_normalization(conv3d(field_input, 1, 48, 3, 1, 'field_conv1'), training=is_train,
                                          name='bn_field1'))
        field_conv2_33 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 3, 1, 'field_conv2_33'), training=is_train,
                                          name='bn_field2_33'))
        field_conv2_11 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 1, 1, 'field_conv2_11'), training=is_train,
                                          name='bn_field2_11'))
        field_conv2_55 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 3, 1, 'field_conv2_55'), training=is_train,
                                          name='bn_field2_55'))
        field_conv2_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv2_55, 16, 16, 3, 1, 'field_conv2_55_1'), training=is_train,
                                          name='bn_field2_55_1'))
        field_cat1 = tf.concat([field_conv2_33, field_conv2_11, field_conv2_55_1], 4)

        field_conv3_33 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 3, 1, 'field_conv3_33'), training=is_train,
                                          name='bn_field3_33'))
        field_conv3_11 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 1, 1, 'field_conv3_11'), training=is_train,
                                          name='bn_field3_11'))
        field_conv3_55 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 3, 1, 'field_conv3_55'), training=is_train,
                                          name='bn_field3_55'))
        field_conv3_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv3_55, 16, 16, 3, 1, 'field_conv3_55_1'), training=is_train,
                                          name='bn_field3_55_1'))
        field_cat2 = tf.concat([field_conv3_33, field_conv3_11, field_conv3_55_1], 4)

        field_conv4_33 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 3, 1, 'field_conv4_33'), training=is_train,
                                          name='bn_field4_33'))
        field_conv4_11 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 1, 1, 'field_conv4_11'), training=is_train,
                                          name='bn_field4_11'))
        field_conv4_55 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 3, 1, 'field_conv4_55'), training=is_train,
                                          name='bn_field4_55'))
        field_conv4_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv4_55, 16, 16, 3, 1, 'field_conv4_55_1'), training=is_train,
                                          name='bn_field4_55_1'))
        field_cat3 = tf.concat([field_conv4_33, field_conv4_11, field_conv4_55_1], 4)

        field_concat = tf.concat([field_cat1, field_cat2, field_cat3], 4)
        field_weight = tf.get_variable('field_w', shape=[144], dtype=tf.float32, initializer=tf.ones_initializer(),
                                       regularizer=None, trainable=True, collections=None)

        field_weight_input = relu(
            tf.layers.batch_normalization(conv3d(field_concat * field_weight, 144, 32, 1, 1, 'field_weight_input'),
                                          training=is_train,
                                          name='bn_field_weight_input'))

        encode1_1 = relu(
            tf.layers.batch_normalization(conv3d(field_weight_input, 32, 32, 3, 1, 'encode1_1'), training=is_train,
                                          name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=is_train,
                                                       name='bnen1_2'))

        encode2_1 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 2, 'encode2_1'), training=is_train,
                                                       name='bnencode2_1'))
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=is_train,
                                                       name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=is_train,
                                                       name='bnen2_3'))

        encode3_1 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 2, 'encode3_1'), training=is_train,
                                                       name='bnencode3_1'))
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=is_train,
                                                       name='bnen3_2'))
        encode3_3 = relu(
            tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=is_train,
                                          name='bnen3_3'))

        encode4_1 = relu(
            tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 2, 'encode4_1'), training=is_train,
                                          name='bnencode4_1'))
        encode4_2 = relu(
            tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_2'), training=is_train,
                                          name='bnen4_2'))

        encode4_3 = relu(
            tf.layers.batch_normalization(conv3d(encode4_2, 256, 256, 3, 1, 'encode4_3'), training=is_train,
                                          name='bnen4_3'))
        encode4_4 = relu(
            tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=is_train,
                                          name='bnen4_4'))
        SAM_input1 = tf.concat([field_input, mag_input], 4)


        decode3_1 = relu(deconv3d(encode4_4, 256, 128, 3, 2, 'decode3_1'))
        decode3_2 = tf.concat([decode3_1, encode3_3], 4)
        decode3_4 = relu(
            tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=is_train,
                                          name='bnde3_4'))

        decode3_6 = relu(
            tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_6') ,
                                          training=is_train, name='bnde3_6'))

        decode4_1 = relu(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'))
        decode4_2 = tf.concat([decode4_1, encode2_3], 4)
        decode4_4 = relu(
            tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=is_train,
                                          name='bnde4_4'))

        decode4_6 = relu(
            tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_6') , training=is_train,
                                          name='bnde4_6'))

        decode5_1 = relu(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'))
        decode5_2 = tf.concat([decode5_1, encode1_2], 4)
        decode5_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=is_train,
                                          name='bnde5_4'), SAM_input1, 32, 'addecode5_4'))

        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_6'), training=is_train,
                                          name='bnde5_6')+ decode5_4)
        cosout = conv3d_b(decode5_6, 32, 1, 1, 1, 'out')
        return cosout
def SAQSM_0SAM(field_input,is_train):
    with tf.variable_scope('syn'):
        field_conv1 = relu(
            tf.layers.batch_normalization(conv3d(field_input, 1, 48, 3, 1, 'field_conv1'), training=is_train,
                                          name='bn_field1'))
        field_conv2_33 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 3, 1, 'field_conv2_33'), training=is_train,
                                          name='bn_field2_33'))
        field_conv2_11 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 1, 1, 'field_conv2_11'), training=is_train,
                                          name='bn_field2_11'))
        field_conv2_55 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 3, 1, 'field_conv2_55'), training=is_train,
                                          name='bn_field2_55'))
        field_conv2_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv2_55, 16, 16, 3, 1, 'field_conv2_55_1'), training=is_train,
                                          name='bn_field2_55_1'))
        field_cat1 = tf.concat([field_conv2_33, field_conv2_11, field_conv2_55_1], 4)

        field_conv3_33 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 3, 1, 'field_conv3_33'), training=is_train,
                                          name='bn_field3_33'))
        field_conv3_11 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 1, 1, 'field_conv3_11'), training=is_train,
                                          name='bn_field3_11'))
        field_conv3_55 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 3, 1, 'field_conv3_55'), training=is_train,
                                          name='bn_field3_55'))
        field_conv3_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv3_55, 16, 16, 3, 1, 'field_conv3_55_1'), training=is_train,
                                          name='bn_field3_55_1'))
        field_cat2 = tf.concat([field_conv3_33, field_conv3_11, field_conv3_55_1], 4)

        field_conv4_33 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 3, 1, 'field_conv4_33'), training=is_train,
                                          name='bn_field4_33'))
        field_conv4_11 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 1, 1, 'field_conv4_11'), training=is_train,
                                          name='bn_field4_11'))
        field_conv4_55 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 3, 1, 'field_conv4_55'), training=is_train,
                                          name='bn_field4_55'))
        field_conv4_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv4_55, 16, 16, 3, 1, 'field_conv4_55_1'), training=is_train,
                                          name='bn_field4_55_1'))
        field_cat3 = tf.concat([field_conv4_33, field_conv4_11, field_conv4_55_1], 4)

        field_concat = tf.concat([field_cat1, field_cat2, field_cat3], 4)
        field_weight = tf.get_variable('field_w', shape=[144], dtype=tf.float32, initializer=tf.ones_initializer(),
                                       regularizer=None, trainable=True, collections=None)

        field_weight_input = relu(
            tf.layers.batch_normalization(conv3d(field_concat * field_weight, 144, 32, 1, 1, 'field_weight_input'),
                                          training=is_train,
                                          name='bn_field_weight_input'))

        encode1_1 = relu(
            tf.layers.batch_normalization(conv3d(field_weight_input, 32, 32, 3, 1, 'encode1_1'), training=is_train,
                                          name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=is_train,
                                                       name='bnen1_2'))

        encode2_1 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 2, 'encode2_1'), training=is_train,
                                                       name='bnencode2_1'))
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=is_train,
                                                       name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=is_train,
                                                       name='bnen2_3'))

        encode3_1 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 2, 'encode3_1'), training=is_train,
                                                       name='bnencode3_1'))
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=is_train,
                                                       name='bnen3_2'))
        encode3_3 = relu(
            tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=is_train,
                                          name='bnen3_3'))

        encode4_1 = relu(
            tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 2, 'encode4_1'), training=is_train,
                                          name='bnencode4_1'))
        encode4_2 = relu(
            tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_2'), training=is_train,
                                          name='bnen4_2'))

        encode4_3 = relu(
            tf.layers.batch_normalization(conv3d(encode4_2, 256, 256, 3, 1, 'encode4_3'), training=is_train,
                                          name='bnen4_3'))
        encode4_4 = relu(
            tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=is_train,
                                          name='bnen4_4'))

        decode3_1 = relu(deconv3d(encode4_4, 256, 128, 3, 2, 'decode3_1'))
        decode3_2 = tf.concat([decode3_1, encode3_3], 4)
        decode3_4 = relu(
            tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=is_train,
                                          name='bnde3_4'))

        decode3_6 = relu(
            tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_6') ,
                                          training=is_train, name='bnde3_6'))

        decode4_1 = relu(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'))
        decode4_2 = tf.concat([decode4_1, encode2_3], 4)
        decode4_4 = relu(
            tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=is_train,
                                          name='bnde4_4'))

        decode4_6 = relu(
            tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_6') , training=is_train,
                                          name='bnde4_6'))

        decode5_1 = relu(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'))
        decode5_2 = tf.concat([decode5_1, encode1_2], 4)
        decode5_4 = relu(
            tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=is_train,
                                          name='bnde5_4'))
        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_6'), training=is_train,
                                          name='bnde5_6'))
        cosout = conv3d_b(decode5_6, 32, 1, 1, 1, 'out')
        return cosout
def SAQSM_NoRB(mag_input,field_input,is_train):
    with tf.variable_scope('syn'):
        field_conv1 = relu(
            tf.layers.batch_normalization(conv3d(field_input, 1, 48, 3, 1, 'field_conv1'), training=is_train,
                                          name='bn_field1'))
        field_conv2_33 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 3, 1, 'field_conv2_33'), training=is_train,
                                          name='bn_field2_33'))
        field_conv2_11 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 1, 1, 'field_conv2_11'), training=is_train,
                                          name='bn_field2_11'))
        field_conv2_55 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 3, 1, 'field_conv2_55'), training=is_train,
                                          name='bn_field2_55'))
        field_conv2_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv2_55, 16, 16, 3, 1, 'field_conv2_55_1'), training=is_train,
                                          name='bn_field2_55_1'))
        field_cat1 = tf.concat([field_conv2_33, field_conv2_11, field_conv2_55_1], 4)

        field_conv3_33 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 3, 1, 'field_conv3_33'), training=is_train,
                                          name='bn_field3_33'))
        field_conv3_11 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 1, 1, 'field_conv3_11'), training=is_train,
                                          name='bn_field3_11'))
        field_conv3_55 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 3, 1, 'field_conv3_55'), training=is_train,
                                          name='bn_field3_55'))
        field_conv3_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv3_55, 16, 16, 3, 1, 'field_conv3_55_1'), training=is_train,
                                          name='bn_field3_55_1'))
        field_cat2 = tf.concat([field_conv3_33, field_conv3_11, field_conv3_55_1], 4)

        field_conv4_33 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 3, 1, 'field_conv4_33'), training=is_train,
                                          name='bn_field4_33'))
        field_conv4_11 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 1, 1, 'field_conv4_11'), training=is_train,
                                          name='bn_field4_11'))
        field_conv4_55 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 3, 1, 'field_conv4_55'), training=is_train,
                                          name='bn_field4_55'))
        field_conv4_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv4_55, 16, 16, 3, 1, 'field_conv4_55_1'), training=is_train,
                                          name='bn_field4_55_1'))
        field_cat3 = tf.concat([field_conv4_33, field_conv4_11, field_conv4_55_1], 4)

        field_concat = tf.concat([field_cat1, field_cat2, field_cat3], 4)
        field_weight = tf.get_variable('field_w', shape=[144], dtype=tf.float32, initializer=tf.ones_initializer(),
                                       regularizer=None, trainable=True, collections=None)

        field_weight_input = relu(
            tf.layers.batch_normalization(conv3d(field_concat * field_weight, 144, 32, 1, 1, 'field_weight_input'),
                                          training=is_train,
                                          name='bn_field_weight_input'))

        encode1_1 = relu(
            tf.layers.batch_normalization(conv3d(field_weight_input, 32, 32, 3, 1, 'encode1_1'), training=is_train,
                                          name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=is_train,
                                                       name='bnen1_2'))

        encode2_1 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 2, 'encode2_1'), training=is_train,
                                                       name='bnencode2_1'))
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=is_train,
                                                       name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=is_train,
                                                       name='bnen2_3'))

        encode3_1 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 2, 'encode3_1'), training=is_train,
                                                       name='bnencode3_1'))
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=is_train,
                                                       name='bnen3_2'))
        encode3_3 = relu(
            tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=is_train,
                                          name='bnen3_3'))

        encode4_1 = relu(
            tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 2, 'encode4_1'), training=is_train,
                                          name='bnencode4_1'))
        encode4_2 = relu(
            tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_2'), training=is_train,
                                          name='bnen4_2'))

        SAM_input1 = tf.concat([field_input, mag_input], 4)
        SAM_input2 = avg_pool3d(SAM_input1)
        SAM_input3 = avg_pool3d(SAM_input2)
        SAM_input4 = avg_pool3d(SAM_input3)
        encode4_4 = relu(SA_norm(tf.layers.batch_normalization(conv3d(encode4_2, 256, 256, 3, 1, 'encode4_4'), training=is_train, name='bnen4_4'), SAM_input4, 256, 'adencode4_4'))
        decode3_1 = relu(deconv3d(encode4_4, 256, 128, 3, 2, 'decode3_1'))
        decode3_2 = tf.concat([decode3_1, encode3_3], 4)
        decode3_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=is_train,
                                          name='bnde3_4'), SAM_input3, 128, 'addecode3_4'))

        decode4_1 = relu(
            deconv3d(decode3_4, 128, 64, 3, 2, 'decode2_1'))
        decode4_2 = tf.concat([decode4_1, encode2_3], 4)
        decode4_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=is_train,
                                          name='bnde4_4'), SAM_input2, 64, 'addecode4_4'))

        decode5_1 = relu(
            deconv3d(decode4_4, 64, 32, 3, 2, 'decode1_1'))
        decode5_2 = tf.concat([decode5_1, encode1_2], 4)
        decode5_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=is_train,
                                          name='bnde5_4'), SAM_input1, 32, 'addecode5_4'))

        cosout = conv3d_b(decode5_4, 32, 1, 1, 1, 'out')
        return cosout
def SAQSM_NoWeight(mag_input,field_input,is_train):
    with tf.variable_scope('syn'):
        field_conv1 = relu(
            tf.layers.batch_normalization(conv3d(field_input, 1, 48, 3, 1, 'field_conv1'), training=is_train,
                                          name='bn_field1'))
        field_conv2_33 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 3, 1, 'field_conv2_33'), training=is_train,
                                          name='bn_field2_33'))
        field_conv2_11 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 1, 1, 'field_conv2_11'), training=is_train,
                                          name='bn_field2_11'))
        field_conv2_55 = relu(
            tf.layers.batch_normalization(conv3d(field_conv1, 48, 16, 3, 1, 'field_conv2_55'), training=is_train,
                                          name='bn_field2_55'))
        field_conv2_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv2_55, 16, 16, 3, 1, 'field_conv2_55_1'), training=is_train,
                                          name='bn_field2_55_1'))
        field_cat1 = tf.concat([field_conv2_33, field_conv2_11, field_conv2_55_1], 4)

        field_conv3_33 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 3, 1, 'field_conv3_33'), training=is_train,
                                          name='bn_field3_33'))
        field_conv3_11 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 1, 1, 'field_conv3_11'), training=is_train,
                                          name='bn_field3_11'))
        field_conv3_55 = relu(
            tf.layers.batch_normalization(conv3d(field_cat1, 48, 16, 3, 1, 'field_conv3_55'), training=is_train,
                                          name='bn_field3_55'))
        field_conv3_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv3_55, 16, 16, 3, 1, 'field_conv3_55_1'), training=is_train,
                                          name='bn_field3_55_1'))
        field_cat2 = tf.concat([field_conv3_33, field_conv3_11, field_conv3_55_1], 4)

        field_conv4_33 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 3, 1, 'field_conv4_33'), training=is_train,
                                          name='bn_field4_33'))
        field_conv4_11 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 1, 1, 'field_conv4_11'), training=is_train,
                                          name='bn_field4_11'))
        field_conv4_55 = relu(
            tf.layers.batch_normalization(conv3d(field_cat2, 48, 16, 3, 1, 'field_conv4_55'), training=is_train,
                                          name='bn_field4_55'))
        field_conv4_55_1 = relu(
            tf.layers.batch_normalization(conv3d(field_conv4_55, 16, 16, 3, 1, 'field_conv4_55_1'), training=is_train,
                                          name='bn_field4_55_1'))
        field_cat3 = tf.concat([field_conv4_33, field_conv4_11, field_conv4_55_1], 4)

        field_concat = tf.concat([field_cat1, field_cat2, field_cat3], 4)

        field_NoWeight_input = relu(
            tf.layers.batch_normalization(conv3d(field_concat , 144, 32, 1, 1, 'field_weight_input'),
                                          training=is_train,
                                          name='bn_field_weight_input'))

        encode1_1 = relu(
            tf.layers.batch_normalization(conv3d(field_NoWeight_input, 32, 32, 3, 1, 'encode1_1'), training=is_train,
                                          name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=is_train,
                                                       name='bnen1_2'))

        encode2_1 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 2, 'encode2_1'), training=is_train,
                                                       name='bnencode2_1'))
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=is_train,
                                                       name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=is_train,
                                                       name='bnen2_3'))

        encode3_1 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 2, 'encode3_1'), training=is_train,
                                                       name='bnencode3_1'))
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=is_train,
                                                       name='bnen3_2'))
        encode3_3 = relu(
            tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=is_train,
                                          name='bnen3_3'))

        encode4_1 = relu(
            tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 2, 'encode4_1'), training=is_train,
                                          name='bnencode4_1'))
        encode4_2 = relu(
            tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_2'), training=is_train,
                                          name='bnen4_2'))

        SAM_input1 = tf.concat([field_input, mag_input], 4)
        SAM_input2 = avg_pool3d(SAM_input1)
        SAM_input3 = avg_pool3d(SAM_input2)
        SAM_input4 = avg_pool3d(SAM_input3)
        encode4_4 = relu(SA_norm_NoWeight(tf.layers.batch_normalization(conv3d(encode4_2, 256, 256, 3, 1, 'encode4_4'),
                                        training=is_train, name='bnen4_4'), SAM_input4, 256, 'adencode4_4'))

        encode4_5 = relu(
            tf.layers.batch_normalization(conv3d(encode4_4, 256, 256, 3, 1, 'encode4_5'), training=is_train,
                                          name='bnen4_5')+encode4_2)

        decode3_1 = relu(tf.layers.batch_normalization(deconv3d(encode4_5, 256, 128, 3, 2, 'decode3_1'), training=is_train,
                                                       name='bnde3_1'))
        decode3_2 = tf.concat([decode3_1, encode3_3], 4)
        decode3_4 = relu(SA_norm_NoWeight(
            tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=is_train,
                                          name='bnde3_4'), SAM_input3, 128, 'addecode3_4'))

        decode3_6 = relu(
            tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_6') ,
                                          training=is_train, name='bnde3_6')+decode3_4)

        decode4_1 = relu(
            tf.layers.batch_normalization(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'), training=is_train,
                                                  name='bnde2_1'))
        decode4_2 = tf.concat([decode4_1, encode2_3], 4)
        decode4_4 = relu(SA_norm_NoWeight(
            tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=is_train,
                                          name='bnde4_4'), SAM_input2, 64, 'addecode4_4'))

        decode4_6 = relu(
            tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_6') , training=is_train,
                                          name='bnde4_6')+ decode4_4)

        decode5_1 = relu(
            tf.layers.batch_normalization(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'), training=is_train,
                                                  name='bnde1_1'))
        decode5_2 = tf.concat([decode5_1, encode1_2], 4)
        decode5_4 = relu(SA_norm_NoWeight(
            tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=is_train,
                                          name='bnde5_4'), SAM_input1, 32, 'addecode5_4'))

        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_6'), training=is_train,
                                          name='bnde5_6')+ decode5_4)

        cosout = conv3d_b(decode5_6, 32, 1, 1, 1, 'out')
        return cosout
def SAQSM_NoIB(mag_input,field_input,is_train):
    with tf.variable_scope('syn'):
        encode1_1=relu(tf.layers.batch_normalization(conv3d(field_input,1,32,3,1,'encode1_1'), training=is_train, name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=is_train, name='bnen1_2'))


        encode2_1=relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 2, 'encode2_1'), training=is_train, name='bnencode2_1'))
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=is_train, name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=is_train, name='bnen2_3'))


        encode3_1=relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 2, 'encode3_1'), training=is_train, name='bnencode3_1'))
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=is_train, name='bnen3_2'))
        encode3_3 = relu(tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=is_train, name='bnen3_3'))


        encode4_1 = relu(tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 2, 'encode4_1'), training=is_train, name='bnencode4_1'))
        encode4_3 = relu(tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_3'), training=is_train, name='bnen4_3'))
        SAM_input1 = tf.concat([field_input, mag_input], 4)
        SAM_input2 = avg_pool3d(SAM_input1)
        SAM_input3 = avg_pool3d(SAM_input2)
        SAM_input4 = avg_pool3d(SAM_input3)
        encode4_4 = relu(SA_norm(tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'),
                                        training=is_train, name='bnen4_4'), SAM_input4, 256, 'adencode4_4'))

        encode4_5 = relu(
            tf.layers.batch_normalization(conv3d(encode4_4, 256, 256, 3, 1, 'encode4_5'), training=is_train,
                                          name='bnen4_5')+encode4_3)

        decode3_1 = relu(tf.layers.batch_normalization(deconv3d(encode4_5, 256, 128, 3, 2, 'decode3_1'), training=is_train,
                                                       name='bnde3_1'))
        decode3_2 = tf.concat([decode3_1, encode3_3], 4)
        decode3_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=is_train,
                                          name='bnde3_4'), SAM_input3, 128, 'addecode3_4'))

        decode3_6 = relu(
            tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_6') ,
                                          training=is_train, name='bnde3_6')+decode3_4)

        decode4_1 = relu(
            tf.layers.batch_normalization(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'), training=is_train,
                                                  name='bnde2_1'))
        decode4_2 = tf.concat([decode4_1, encode2_3], 4)
        decode4_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=is_train,
                                          name='bnde4_4'), SAM_input2, 64, 'addecode4_4'))

        decode4_6 = relu(
            tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_6') , training=is_train,
                                          name='bnde4_6')+ decode4_4)

        decode5_1 = relu(
            tf.layers.batch_normalization(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'), training=is_train,
                                                  name='bnde1_1'))
        decode5_2 = tf.concat([decode5_1, encode1_2], 4)
        decode5_4 = relu(SA_norm(
            tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=is_train,
                                          name='bnde5_4'), SAM_input1, 32, 'addecode5_4'))

        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_6'), training=is_train,
                                          name='bnde5_6')+ decode5_4)
        cosout = conv3d_b(decode5_6, 32, 1, 1, 1, 'out')
        return cosout


