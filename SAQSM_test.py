import numpy as np
import scipy.io as sio
import tensorflow as tf
import model
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True

def reduce_dim(x):
    x = x[0]
    x = x[:, :, :, 0]
    return x

def increase_dim(image):
    image = np.reshape(image, ((1,) + image.shape))
    image = image[:, :, :, :, np.newaxis]
    image = image.astype('float32')
    return image

def load_magimage(path):
    image = sio.loadmat(path)
    image = image['magmap']

    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    image = np.reshape(image, ((1,) + image.shape))
    image = image[:, :, :, :, np.newaxis]
    image = image.astype('float32')
    return image

def load_fieldimage(path):
    image = sio.loadmat(path)
    image = image['field']

    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    image = np.reshape(image, ((1,) + image.shape))
    image = image[:, :, :, :, np.newaxis]
    image = image.astype('float32')
    return image

input_width=224
input_height=224
input_depth=110
f1 = np.zeros((input_depth, input_height, input_width), dtype=np.float32)
m1 = np.zeros((input_depth, input_height, input_width), dtype=np.float32)

f = sio.loadmat('/***/***/***.mat')
f = f['field']
f1[0:input_depth, 0:input_height, 0:input_width] = f

m = sio.loadmat('/***/***/***.mat')
m = m['magmap']
m1[0:input_depth, 0:input_height, 0:input_width] = m

mask = sio.loadmat('/***/***/***.mat')
mask = mask['mask']
output = np.zeros((input_depth, input_height, input_width), dtype=np.float32)

with tf.Graph().as_default():
    with tf.Session().as_default() as sess:

        model_path = "/***/***/***.ckpt-done"

        maginput = tf.placeholder(dtype=tf.float32, shape=[1, 16, input_height, input_width, 1])
        fieldinput = tf.placeholder(dtype=tf.float32, shape=[1, 16, input_height, input_width, 1])

        SAQSM_output = model.SAQSM(maginput, fieldinput, False)
        saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
        saver.restore(sess, model_path)

        for i in range(0, input_depth-16, 16):
            fin = increase_dim(f1[i:i + 16, :, :])
            min = increase_dim(m1[i:i + 16, :, :])

            depth16_output = reduce_dim(sess.run(SAQSM_output, feed_dict={maginput: min, fieldinput: fin}))
            res = depth16_output[0:input_depth, 0:input_height, 0:input_width]
            output[i :i + 16, :, :] = res[0:16, :, :]

        output_transpose = sess.run(tf.transpose(output, [1, 2, 0]))
        final_output = output_transpose * mask
        sio.savemat('/***/***/output.mat', {'output': final_output})







