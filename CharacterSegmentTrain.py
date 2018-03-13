from __future__ import print_function
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import TensorflowUtils as utils
# import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange
import os
import cv2

# import pydevd
# pydevd.settrace('192.168.50.217',port=8888, stdoutToServer=True, stderrToServer=True)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6  # occupy GPU40%
session = tf.Session(config=config)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "80", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "gen_imgs/", "path to dataset")
tf.flags.DEFINE_string("test_data_dir", "test_imgs/", "path to test dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "inference", "Mode train/ test/ inference")


MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE = (1024, 48)

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    with tf.variable_scope("inference"):
        down_w_conv1 = utils.weight_variable([3, 3, 1, 32], name='down_w_conv1')
        down_b1 = utils.bias_variable([32], name='down_b1')
        down_conv1 = tf.nn.relu(utils.conv2d_basic(image, down_w_conv1, down_b1))
        down_pool1 = utils.max_pool_2x2(down_conv1)  # (24, 512, 32)

        down_w_conv2 = utils.weight_variable([3, 3, 32, 64], name='down_w_conv2')
        down_b2 = utils.bias_variable([64], name='down_b2')
        down_conv2 = tf.nn.relu(utils.conv2d_basic(down_pool1, down_w_conv2, down_b2))
        down_pool2 = utils.max_pool_2x2(down_conv2)  # (12, 256, 64)

        down_w_conv3 = utils.weight_variable([3, 3, 64, 128], name='down_w_conv3')
        down_b3 = utils.bias_variable([128], name='down_b3')
        down_conv3 = tf.nn.relu(utils.conv2d_basic(down_pool2, down_w_conv3, down_b3))
        down_pool3 = utils.max_pool_2x2(down_conv3)  # (6, 128, 128)

        down_w_conv4 = utils.weight_variable([3, 3, 128, 256], name='down_w_conv4')
        down_b4 = utils.bias_variable([256], name='down_b4')
        down_conv4 = tf.nn.relu(utils.conv2d_basic(down_pool3, down_w_conv4, down_b4))
        down_pool4 = utils.max_pool_2x2(down_conv4)  # (3, 64, 256)

        down_w_conv5 = utils.weight_variable([3, 3, 256, 512], name='down_w_conv5')
        down_b5 = utils.bias_variable([512], name='down_b5')
        down_conv5 = tf.nn.relu(utils.conv2d_basic(down_pool4, down_w_conv5, down_b5))
        dropout5 = tf.nn.dropout(down_conv5, keep_prob=keep_prob)
        # down_pool5 = utils.max_pool_2x2(dropout5)  # (1, 32, 512)
        down_pool5 = tf.nn.max_pool(dropout5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        # now to upscale to actual image size
        up_w_conv1 = utils.weight_variable([1, 5, 512, 512], name="up_w_conv1")
        up_b1 = utils.bias_variable([512], name="up_b1")
        up_conv1 = tf.nn.relu(utils.conv2d_transpose_strided(down_pool5, up_w_conv1, up_b1, output_shape=[tf.shape(image)[0],1,64,512]))

        up_w_conv2 = utils.weight_variable([1, 5, 256, 512], name="up_w_conv2")
        up_b2 = utils.bias_variable([256], name="up_b2")
        up_conv2 = tf.nn.relu(utils.conv2d_transpose_strided(up_conv1, up_w_conv2, up_b2, output_shape=[tf.shape(image)[0], 1, 128, 256]))

        up_w_conv3 = utils.weight_variable([1, 5, 128, 256], name="up_w_conv3")
        up_b3 = utils.bias_variable([128], name="up_b3")
        up_conv3 = tf.nn.relu(utils.conv2d_transpose_strided(up_conv2, up_w_conv3, up_b3, output_shape=[tf.shape(image)[0], 1, 256, 128]))

        up_w_conv4 = utils.weight_variable([1, 5, 64, 128], name="up_w_conv4")
        up_b4 = utils.bias_variable([64], name="up_b4")
        up_conv4 = tf.nn.relu(utils.conv2d_transpose_strided(up_conv3, up_w_conv4, up_b4, output_shape=[tf.shape(image)[0], 1, 512, 64]))

        up_w_conv5 = utils.weight_variable([1, 5, 1, 64], name="up_w_conv5")
        up_b5 = utils.bias_variable([1], name="up_b5")
        up_conv5 = tf.nn.sigmoid(utils.conv2d_transpose_strided(up_conv4, up_w_conv5, up_b5, output_shape=[tf.shape(image)[0], 1, 1024, 1]))

        annotation_pred = up_conv5 > 0.5

    return annotation_pred, up_conv5


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def _transform(filename):
    image = cv2.imread(filename, 0)
    # if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
    #     image = np.array([image for i in range(3)])
    resize_image = cv2.resize(image, (1024, 48))
    return np.expand_dims(np.array(resize_image) / 255.0, axis=3)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE[1], IMAGE_SIZE[0], 1], name="input_image")
    annotation = tf.placeholder(tf.float32, shape=[None, 1, IMAGE_SIZE[0], 1], name="annotation")
    pred_annotation, logits = inference(image, keep_probability)
    # logits = tf.squeeze(logits, squeeze_dims=[1, 3])
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    # loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
    #                                                                       labels=tf.squeeze(annotation, squeeze_dims=[3]),
    #                                                                       name="entropy")))
    alpha = 0.9
    belta = 0.1
    # one sample: -ylog(y)+-(1-y)log(1-y),    n samples: mean(one sample)
    loss = tf.reduce_mean(tf.add(-alpha*tf.reduce_sum(annotation * tf.log(logits + 1e-9), 1),
                                           -belta*tf.reduce_sum((1 - annotation) * tf.log(1 - logits + 1e-9), 1)))
    tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()


    print("Setting up dataset reader")
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(FLAGS.data_dir)
    validation_dataset_reader = dataset.BatchDatset(FLAGS.test_data_dir, dataset_file='dataset_test.txt')

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    ckpt.model_checkpoint_path = 'logs/model.ckpt-100000'
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "inference":
        path = './real_test_imgs/'
        fnames = os.listdir(path)
        imgs = np.array([_transform(os.path.join(path, elem)) for elem in fnames])
        pred = sess.run(pred_annotation, feed_dict={image: imgs,
                                                    keep_probability: 1.0})  # [80, 1,1024,1]
        pred = np.squeeze(pred, axis=3)
        pred = np.squeeze(pred, axis=1)
        pred = np.asarray(pred, np.int)
        res = []
        for itr in range(len(imgs)):
            im = imgs[itr]
            pre = pred[itr]
            im = 255 * np.squeeze(im, axis=2)
            im = Image.fromarray(im)
            # make sure images are of shape(h,w,3)
            img = im.convert('RGB')
            img.save('result/source_%s.jpg' % str(itr))
            res.append(['source_%s.jpg']+list(pre))
            img_d = ImageDraw.Draw(img)
            x_len, y_len = img.size
            for x in range(x_len):
                if pre[x] == 1:
                    img_d.line(((x, 0), (x, y_len)), (250, 0, 0))
            img.save('result/pred_%s.jpg' % str(itr))
            # utils.save_image(im.astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5 + itr))
        np.savetxt('res.txt', res, fmt='%s')
    elif FLAGS.mode == "test":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images,
                                                    keep_probability: 1.0})  # [80, 1,1024,1]
        pred = np.squeeze(pred, axis=3)
        pred = np.squeeze(pred, axis=1)
        pred = np.asarray(pred, np.int)
        for itr in range(FLAGS.batch_size):
            im = valid_images[itr]
            pre = pred[itr]
            im = 255*np.squeeze(im,axis=2)
            im = Image.fromarray(im)
            # make sure images are of shape(h,w,3)
            img = im.convert('RGB')

            img_d = ImageDraw.Draw(img)
            x_len, y_len = img.size
            for x in range(x_len):
                if pre[x] == 1:
                    img_d.line(((x,0),(x,y_len)),(250,0,0))
            img.save('result/pred_%s.jpg' % str(itr))
            # utils.save_image(im.astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5 + itr))

if __name__ == "__main__":
    tf.app.run()
