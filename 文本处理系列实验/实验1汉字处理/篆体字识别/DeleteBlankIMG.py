import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def clean_blank():
    img_dir = "data"  # 设置待清除空白图片的文件夹路径
    for each_cls in tqdm(os.listdir(img_dir), desc='正在清除空白图片'):
        dir_path = os.path.join(img_dir, each_cls)
        for each_img in os.listdir(dir_path):
            image_path = os.path.join(dir_path, each_img)
            # 加载图片，并将图片转成ndarray类型
            img_array = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(image_path))
            # 判断每张图片是否是空白图片，若是，则删除。
            if np.all(img_array == 255):
                os.remove(image_path)


if __name__ == '__main__':
    clean_blank()
