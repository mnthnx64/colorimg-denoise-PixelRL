import os
import numpy as np
import cv2
import math
 
"""
Used to read in images for testing
"""

class TestBatchLoader(object):
 
    def __init__(self, img, crop_size, validation=False):
        self.img = img = cv2.imread(img,1)
        self.crop_size = crop_size
 
    # test ok
    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path
 
    # test ok
    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c
 
    # test ok
    @staticmethod
    def read_paths(txt_path, src_path):
        cs = []
        for pair in TestBatchLoader.path_label_generator(txt_path, src_path):
            cs.append(pair)
        return cs
 
    def load_training_data(self, indices):
        return self.load_data()
 
    def load_testing_data(self, indices):
        return self.load_data()

    def load_validation_data(self):
        return self.load_data()
 
    # test ok
    def load_data(self):
        in_channels = 3

        img = self.img

        if img is None:
            raise RuntimeError("invalid image")

        h, w, c = img.shape

        self.ratioh  = math.ceil(h/self.crop_size)
        self.ratiow = math.ceil(w/self.crop_size)

        self.extra_h = self.ratioh*self.crop_size - h
        self.extra_w = self.ratiow*self.crop_size - w

        xs = np.zeros((self.ratioh*self.ratiow, in_channels, self.crop_size, self.crop_size)).astype(np.float32)

        img_extrah = np.zeros(shape=(self.extra_h, w, 3))
        img_extraw = np.zeros(shape=(h+self.extra_h, self.extra_w, 3))

        cut_img = np.concatenate((img,img_extrah),axis = 0)
        cut_img = np.concatenate((cut_img,img_extraw),axis = 1)

        for i in range(self.ratiow):
            for j in range(self.ratioh):
                xs[j+self.ratioh*i, :, :, :] = (cut_img[0+j*self.crop_size :self.crop_size +j*self.crop_size ,0+i*self.crop_size :self.crop_size +i*self.crop_size ]/255).astype(np.float32).reshape(c,self.crop_size,self.crop_size)

        return xs

    def stitch_image(self, batch):
        all_img = []
        for i in range(self.ratiow):
            col_img = batch[self.ratioh*i].reshape(self.crop_size,self.crop_size,3)
            for j in range(1, self.ratioh):
                col_img = np.concatenate((col_img, batch[j+self.ratioh*i].reshape(self.crop_size,self.crop_size,3)), axis = 0)
            all_img.append(col_img)
        
        st_img = all_img[0]
        for x in range(1, len(all_img)):
            st_img = np.concatenate((st_img, all_img[x]), axis = 1)

        return st_img[:-self.extra_h,:-self.extra_w]
