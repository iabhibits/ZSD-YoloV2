import os
import glob
import argparse
import pickle
import cv2
import numpy as np
from embedding.embed_utils import *
from embedding.yolo_net import Yolo

#os.environ["CUDA_VISIBLE_DEVICES"]="1"
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor']


def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="model")
    parser.add_argument("--pre_trained_model_path", type=str, default="../trained_models/whole_model_trained_yolo_voc")
    parser.add_argument("--input", type=str, default="../mydata2/1505split/seen.txt")
    parser.add_argument("--output", type=str, default="test_images")

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path)
        else:
            model = Yolo(20)
            model.load_state_dict(torch.load(opt.pre_trained_model_path))
    else:
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage)
        else:
            model = Yolo(20)
            model.load_state_dict(torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage))
    model.eval()
    colors = pickle.load(open("src/pallete", "rb"))

    embeddings = []

    file = open(opt.input, "r")
    content_list = []
    for line in file:
        content_list.append(line.rstrip('\n'))
    file.close()

    for image_path in content_list:
        if "prediction" in image_path:
            continue
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = cv2.resize(image, (opt.image_size, opt.image_size))
        image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        image = image[None, :, :, :]
        width_ratio = float(opt.image_size) / width
        height_ratio = float(opt.image_size) / height
        data = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            data = data.cuda()
        with torch.no_grad():
            logits = model(data)
            predictions = gcn_post_processing(logits, opt.image_size, CLASSES, model.anchors, opt.conf_threshold,
                                          opt.nms_threshold)
            embed, label = predictions
            label = label.detach().cpu().numpy()
            embed = embed.detach().cpu().numpy()
            embeddings.append([label,embed])

    with open('sample','wb') as f:
        pickle.dump(embeddings,f)

if __name__ == "__main__":
    opt = get_args()
    test(opt)
