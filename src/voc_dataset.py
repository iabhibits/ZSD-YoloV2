"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from src.data_augmentation import *


class VOCZSD_Dataset(Dataset):
    def __init__(self, root_path="data_path", year="2007",split = "1505", mode="seen", image_size=448, is_training = True):
        if (mode in ["mix", "seen", "test_seen", "unseen"] and year == "2007") or (
                mode in ["mix", "seen", "test_seen", "unseen"] and year == "2012"):
            #self.data_path = os.path.join(root_path, "VOC{}".format(year))
            self.data_path = root_path
            self.zsddata_path = os.path.join(root_path, "{}split".format(split))
        id_list_path = os.path.join(self.zsddata_path, "{}.txt".format(mode))
        self.ids = [id.strip() for id in open(id_list_path)]
        # self.seen_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cow',
        #                 'dog', 'horse', 'person', 'potted plant', 'sheep', 'sofa', 'train',
        #                 'tvmonitor']
        # self.unseen_classes = ['bus','cat', 'chair', 'dining table', 'motorbike']
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.ids)
        self.is_training = is_training
        self.mode = mode

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        id = self.ids[item]
        image_path = id
        #print("Image Path {}\n".format(image_path))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_id1 = image_path.split('/')
        image_id1 = image_id1[-1]
        image_id1 = image_id1.strip('.jpg')
        image_id = image_path.strip('.jpg')
        child_dir = os.path.dirname(image_id)
        #print("child dir {}\n".format(child_dir))
        parent_dir = os.path.abspath(os.path.join(child_dir, os.pardir))
        #print("parent directory {}\n".format(parent_dir))

        #data_path = os.path.join(self.data_path, "{}".format(year))
        image_xml_path = os.path.join(parent_dir, "Annotations", "{}.xml".format(image_id1))
        #print("image_xml_path {}\n".format(image_xml_path))
        annot = ET.parse(image_xml_path)

        objects = []
        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            # if self.mode == 'seen':
            #     label = self.seen_classes.index(obj.find('name').text.lower().strip())
            # elif self.mode == 'unseen':
            #     label = self.unseen_classes.index(obj.find('name').text.lower().strip())

            label = self.classes.index(obj.find('name').text.lower().strip())
            objects.append([xmin, ymin, xmax, ymax, label])
        if self.is_training:
            transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.image_size)])
        else:
            transformations = Compose([Resize(self.image_size)])
        image, objects = transformations((image, objects))

        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)
