# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from PIL import Image, ImageFont, ImageDraw
import numpy as np
from matplotlib import pyplot as plt

from skimage import draw

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import json
import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained'] = False
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
predictor_dm = Predictor(config)


def crop_sub_image(image, xy1, xy2, xy3, xy4):
    """
    Crop sub by 4 corner of image
    """
    temp_img = image.copy()
    mask = np.zeros_like(temp_img)
    x1 = min(xy1[0], xy2[0], xy3[0], xy4[0])
    y1 = min(xy1[1], xy2[1], xy3[1], xy4[1])
    x2 = max(xy1[0], xy2[0], xy3[0], xy4[0])
    y2 = max(xy1[1], xy2[1], xy3[1], xy4[1])
    # tmp = image[y1:y2, x1:x2]
    # cv2.fillPoly(mask, poly=)
    xs, ys = zip(*[xy1, xy2, xy3, xy4])
    fill_row, fill_col = draw.polygon(ys, xs, image.shape)
    mask[fill_row, fill_col]= 1
    tmp_img = mask * temp_img

    tmp = tmp_img[y1:y2, x1:x2]
    # plt.imshow(tmp)
    # plt.show()
    return tmp.copy()






def draw_det_res(dt_boxes, config, img, img_name, save_path):
    if len(dt_boxes) > 0:
        import cv2
        src_im = img
        img_name_temp = img_name.split('/')[-1]
        f = open(f"out/{img_name_temp}.txt", "w")
        for box in dt_boxes:
            sub_img = crop_sub_image(src_im, box[0], box[1], box[2], box[3])
            sentence = predictor_dm.predict(Image.fromarray(sub_img))
            box = box.astype(np.int32).reshape((-1, 1, 2))
            # print(box.shape)
            f.write(f"{box[0][0][0]},{box[0][0][1]},{box[1][0][0]},{box[1][0][1]},{box[2][0][0]},{box[2][0][1]},{box[3][0][0]},{box[3][0][1]}\n")
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
            #write viet name text
            fontpath = "/Users/phanmanhtuan/Downloads/font-times-new-roman/dmm.ttf" # <== 这里是宋体路径
            fontpath = "Times.ttc"
            font = ImageFont.truetype(fontpath, int(abs((box[0][0][1] - box[3][0][1]) * 0.5)))
            img_de_ve = Image.fromarray(src_im.copy())
            # cv2.putText(src_im, sentence, (box[0][0][0], box[0][0][1]), cv2.FONT_ITALIC, 1, (255, 255, 0), 1)
            draw = ImageDraw.Draw(img_de_ve)
            draw.text((box[0][0][0], box[0][0][1]), sentence, font=font, fill=(0, 0, 255))
            # draw.text((10,10), sentence, font=font, fill=(255, 255, 0))
            # plt.imshow(img_de_ve)
            # plt.show()
            src_im = np.array(img_de_ve)


        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, os.path.basename(img_name))
        cv2.imwrite(save_path, src_im)
        logger.info("The detected Image saved in {}".format(save_path))
        f.close()


@paddle.no_grad()
def main():
    global_config = config['Global']

    # build model
    model = build_model(config['Architecture'])

    load_model(config, model)
    # build post process
    post_process_class = build_post_process(config['PostProcess'])

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)

    ops = create_operators(transforms, global_config)

    save_res_path = config['Global']['save_res_path']
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()
    with open(save_res_path, "wb") as fout:
        for file in get_image_file_list(config['Global']['infer_img']):
            logger.info("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = paddle.to_tensor(images)
            preds = model(images)
            post_result = post_process_class(preds, shape_list)

            src_img = cv2.imread(file)

            dt_boxes_json = []
            # parser boxes if post_result is dict
            if isinstance(post_result, dict):
                det_box_json = {}
                for k in post_result.keys():
                    boxes = post_result[k][0]['points']
                    dt_boxes_list = []
                    for box in boxes:
                        tmp_json = {"transcription": ""}
                        tmp_json['points'] = box.tolist()
                        dt_boxes_list.append(tmp_json)
                    det_box_json[k] = dt_boxes_list
                    save_det_path = os.path.dirname(config['Global'][
                        'save_res_path']) + "/det_results_{}/".format(k)
                    draw_det_res(boxes, config, src_img, file, save_det_path)
            else:
                boxes = post_result[0]['points']
                dt_boxes_json = []
                # write result
                for box in boxes:
                    tmp_json = {"transcription": ""}
                    tmp_json['points'] = box.tolist()
                    dt_boxes_json.append(tmp_json)
                save_det_path = os.path.dirname(config['Global'][
                    'save_res_path']) + "/det_results/"
                draw_det_res(boxes, config, src_img, file, save_det_path)
            otstr = file + "\t" + json.dumps(dt_boxes_json) + "\n"
            fout.write(otstr.encode())

    logger.info("success!")


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()