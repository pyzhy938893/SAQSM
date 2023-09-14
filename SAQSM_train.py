import numpy as np
import scipy.io as sio
import model
import tensorflow as tf
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
def load_magimage(path):
    image = sio.loadmat(path)
    image = image['magmap']
    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    image = np.reshape(image, ((1,) + image.shape))
    image = image[:,:,  :, :, np.newaxis]
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

def load_cosmos(path):
    image = sio.loadmat(path)
    image = image['chi_mo']
    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    image = np.reshape(image, ((1,) + image.shape))
    image = image[:,  :, :,:, np.newaxis]
    image = image.astype('float32')
    return image

epochnum =20
batch_size=4
height=64
width=64
depth=16
model_path =('/***/***/***')

maginput = tf.placeholder(dtype=tf.float32)
fieldinput = tf.placeholder(dtype=tf.float32)

coslabel = tf.placeholder(dtype=tf.float32, shape=[batch_size, depth,height, width, 1])

out= model.SAQSM(maginput,fieldinput,True)
gen_loss=model.mse_gdl_loss(out,coslabel)

variable_to_ref = []
for variable in tf.trainable_variables():
    if (variable.name.startswith('syn')):
        variable_to_ref.append(variable)

update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    gen_op = tf.train.AdamOptimizer(0.001).minimize(gen_loss, var_list=variable_to_ref)

variables_to_restore = []
for v in tf.global_variables():
    variables_to_restore.append(v)
saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V1, max_to_keep=None)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    last_file = tf.train.latest_checkpoint(model_path)
    if last_file:
        tf.logging.info('Restoring model from {}'.format(last_file))
        saver.restore(sess, last_file)

    for epoch in range(epochnum):
        sumnum = []
        patches_num=28800 #total 28800 pairs of patches

        for i in range(1, patches_num):
            sumnum.append(i)
        for x in range(1,patches_num//4):
            sample=random.sample(sumnum,batch_size)
            sumnum.remove(sample[0])
            sumnum.remove(sample[1])
            sumnum.remove(sample[2])
            sumnum.remove(sample[3])
            i1 = sample[0]
            i2 = sample[1]
            i3 = sample[2]
            i4 = sample[3]

            mimg1 = model.load_magimage('/***/***/mag%d.mat' % i1)
            mimg2 = model.load_magimage('/***/***/mag%d.mat' % i2)
            mimg3 = model.load_magimage('/***/***/mag%d.mat' % i3)
            mimg4 = model.load_magimage('/***/***/mag%d.mat' % i4)
            _maginput = np.concatenate([mimg1, mimg2, mimg3, mimg4], 0)

            fimg1 =  model.load_fieldimage('/***/***/field%d.mat' % i1)
            fimg2 =  model.load_fieldimage('/***/***/field%d.mat' % i2)
            fimg3 =  model.load_fieldimage('/***/***/field%d.mat' % i3)
            fimg4 =  model.load_fieldimage('/***/***/field%d.mat' % i4)
            _fieldinput = np.concatenate([fimg1, fimg2, fimg3, fimg4], 0)

            cimg1 = model.load_cosmos('/***/***/cos%d.mat' % i1)
            cimg2 = model.load_cosmos('/***/***/cos%d.mat' % i2)
            cimg3 = model.load_cosmos('/***/***/cos%d.mat' % i3)
            cimg4 = model.load_cosmos('/***/***/cos%d.mat' % i4)
            _cosinput = np.concatenate([cimg1, cimg2, cimg3, cimg4], 0)

            sess.run(gen_op,
                     feed_dict={maginput: _maginput, fieldinput: _fieldinput,coslabel:_cosinput})
            if x % 40 == 0:
                genloss_1 = sess.run(gen_loss,
                                  feed_dict={maginput: _maginput, fieldinput: _fieldinput,
                                             coslabel: _cosinput})
                print('costepoch%d:' % x, genloss_1)

        if  epoch %1 == 0:
            saver.save(sess, os.path.join(model_path, 'saqsm_epoch_%d.ckpt-done' % epoch))
            tf.logging.info('Done training%d -- epoch limit reached' % epoch)




