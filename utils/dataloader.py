import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input


def parse_annotation_from_num(xml_dir=r"C:\Users\ASUS\Desktop\深度学习\fast_rcnn\voc2007\VOCdevkit\VOC2007\Annotations",
                              image_dir=r"C:\Users\ASUS\Desktop\深度学习\fast_rcnn\voc2007\VOCdevkit\VOC2007\JPEGImages",
                              annotation_lines=None):
    """
    根据 annotation_lines 中的数字 num，解析 XML 文件获取标注信息。
    :param xml_dir: XML 文件夹路径
    :param image_dir: 图像文件夹路径
    :param annotation_lines: 仅包含图片编号的列表
    :return: 解析后的 annotation_lines，结构为 (batch, 1, (num.jpg, x1, y1, x2, y2, ...))
    """
    parsed_lines = []
    for nu in annotation_lines:
        # 找到对应的 XML 文件
        num=nu.strip()
        xml_file = os.path.join(xml_dir, f"{num}.xml")
        if not os.path.exists(xml_file):
            raise FileNotFoundError(f"XML file not found: {xml_file}")

        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 获取图像文件名
        image_name = root.find("filename").text
        image_path = os.path.join(image_dir, image_name)

        # 提取目标框和类别
        objects = root.findall("object")
        bbox_info = []
        for obj in objects:
            bbox = obj.find("bndbox")
            x_min = int(bbox.find("xmin").text)
            y_min = int(bbox.find("ymin").text)
            x_max = int(bbox.find("xmax").text)
            y_max = int(bbox.find("ymax").text)
            bbox_info.extend([x_min, y_min, x_max, y_max])

        # 构建解析行
        parsed_lines.append((image_path, *bbox_info))

    return parsed_lines


class faster_rcnn_dataloader(Dataset):

    def __init__(self, annotation_lines, input_shape=[600, 600], train=True):
        """
        初始化 Faster R-CNN 数据加载器
        :param annotation_lines: 包含 (num.jpg, x1, y1, x2, y2, ...) 格式的注释列表
        :param input_shape: 网络输入大小
        :param train: 是否为训练模式
        """
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        annotation_line = self.annotation_lines[index]

        # 拆分图片路径和标注框
        image_path = annotation_line[0]
        box = np.array(annotation_line[1:]).reshape(-1, 4)

        # 读取图像并随机增强
        image, y = self.get_random_data(image_path, box, self.input_shape[0:2], random=self.train)

        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box_data = np.zeros((len(y), 5))
        if len(y) > 0:
            box_data[:len(y),:4] = y

        box = box_data[:, :4]
        label = box_data[:, -1]
        return image, box, label

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image_path, box, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        """
        对图像和标注框进行随机增强处理
        :param image_path: 图像路径
        :param box: 标注框数据
        :param input_shape: 网络输入大小
        :param jitter: 随机抖动范围
        :param hue: 色调变换范围
        :param sat: 饱和度变换范围
        :param val: 亮度变换范围
        :param random: 是否进行随机增强
        """
        image = Image.open(image_path)
        image = cvtColor(image)

        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # 调整图像大小
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # 调整标注框
            if len(box) > 0:
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框

            return image_data, box

        # 随机增强开始
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 在图像上填充灰条
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 随机翻转
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)

        # HSV 色域变换
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # 调整标注框
        if len(box) > 0:
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # 过滤无效框

        return image_data, box

        # 随机增强代码省略...


def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = torch.from_numpy(np.array(images))
    return images, bboxes, labels
