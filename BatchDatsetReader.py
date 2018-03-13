"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
# import scipy.misc as misc
import cv2
import os


class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, datadir='gen_imgs', dataset_file='dataset.txt', image_options={'resize': True, 'resize_size': (1024, 48)}):
        """
        Intialize a generic file reader with batching for list of files
        :param dataset_file: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")

        print(image_options)
        f = open(dataset_file, 'r')
        self.files = f.readlines()
        self.image_options = image_options
        self.datadir = datadir
        self._read_images()

    def _read_images(self):
        self.images = np.array([eval(filename)[0] for filename in self.files])
        self.annotations = np.array([eval(filename)[1:] for filename in self.files])
        print (self.images.shape)
        print (self.annotations.shape)

    def _transform(self, filename):
        
        image = cv2.imread(filename, 0)
        if image is None:
            return None
        # if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
        #     image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = self.image_options["resize_size"]
            resize_image = cv2.resize(image, resize_size)
        else:
            resize_image = image

        return np.expand_dims(np.array(resize_image)/255.0, axis=3)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > len(self.images):
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(len(self.images))
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        im_names = self.images[start:end]
        arr = []
        for elem in im_names:
            tmp = self._transform(os.path.join(self.datadir, elem))
            if tmp is None:
                continue
            arr.append(tmp)
        imgs = np.array(arr)
        # imgs = np.array([self._transform(os.path.join(self.datadir, elem)) for elem in im_names])
        annotations = self.annotations[start:end]
        labels = np.zeros((len(annotations), self.image_options["resize_size"][0]))
        for i in range(len(annotations)):
            labels[i][annotations[i]] = 1
        labels = np.expand_dims(labels, axis=1)  # [80,1, 1024]
        labels = np.expand_dims(labels, axis=3)  # [80,1, 1024,1]
        return imgs, labels

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, len(self.images), size=[batch_size]).tolist()
        im_names = self.images[indexes]
        arr = []
        for elem in im_names:
            tmp = self._transform(os.path.join(self.datadir, elem))
            if tmp is None:
                continue
            arr.append(tmp)
        # imgs = np.array([self._transform(os.path.join(self.datadir, elem)) for elem in im_names])
        imgs = np.array(arr)
        annotations = self.annotations[indexes]
        labels = np.zeros((len(annotations), self.image_options["resize_size"][0]))
        for i in range(len(annotations)):
            labels[i][annotations[i]] = 1
        labels = np.expand_dims(labels, axis=1)  # [80,1, 1024]
        labels = np.expand_dims(labels, axis=3)  # [80,1, 1024,1]
        return imgs, labels


# data = BatchDatset()
# a = data.next_batch(3)
# print a
