import cv2
import os
import glob
import numpy as np
import json
import shutil
from tqdm import tqdm


def categorie():
    categorie={}
        # categorie['supercategory'] = label[0]
        # categorie['id']=len(self.label)+1 # 0 默认为背景
        # categorie['name'] = label[1]
    categorie['supercategory'] = 'hand'
    categorie['id'] = 1  # 0 默认为背景
    categorie['name'] = 'hand'
    categorie['keypoints'] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
    categorie['skeleton'] = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
    return categorie


def data2coco(img_list, categorie, annotation):
    data_coco={}
    data_coco['images']=img_list
    data_coco['annotations']=annotation
    data_coco['categories']=[categorie]
    return data_coco


json_list = glob.glob("F:/1228_data_gesture\json_all/*.json")
count = 0
img_list=[]
anno_list = []
for json_file in tqdm(json_list):

    with open(json_file, 'r') as f:
        ann_data = json.load(f)

    img = img = cv2.imread("F:/1228_data_gesture/images_all/" + ann_data['image_file'])
    if img is None:
        continue
    # cv2.imwrite(dst, img)
    # cv2.imwrite(dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

    img_height = img.shape[0]
    img_width = img.shape[1]

    h, w = img.shape[:2]
    keypoints = np.array(ann_data['keypoints'])
    xs = keypoints[:, 0] * w
    ys = keypoints[:, 1] * h

    coord = []
    kpt = []

    for i in range(21):
        co_x = xs[i]
        co_y = ys[i]
        coord.append([int(co_x+0.5), int(co_y+0.5)])
        kpt.append(int(co_x+0.5))
        kpt.append(int(co_y+0.5))
        kpt.append(2)

    coord_np = np.array(coord)
    x_min = np.min(coord_np[:, 0])
    y_min = np.min(coord_np[:, 1])
    x_max = np.max(coord_np[:, 0])
    y_max = np.max(coord_np[:, 1])
    w_o = x_max - x_min
    h_o = y_max - y_min
    scale = 0.5
    # 中指指尖长度
    l_ = np.sqrt((coord_np[10][0] - coord_np[11][0])**2 + (coord_np[10][1] - coord_np[11][1])**2)
    x1 = max(0, x_min - scale * l_)
    y1 = max(0, y_min - scale * l_)
    x2 = min(img_width, x_max + scale * l_)
    y2 = min(img_height, y_max + scale * l_)


    image = {}
    image['file_name'] = ann_data['image_file']  # imgPath
    image['height'] = img_height
    image['width'] = img_width
    image['id'] = count
    img_list.append(image)

    annotation = {}
    annotation['iscrowd'] = 0
    annotation['image_id'] = image['id']
    annotation['area'] = (x2 - x1) * (y2 - y1)  # self.ComputePolygonArea(list(np.asarray(points).flatten()))
    annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]  #x1, y1 ,w, h
    annotation['num_keypoints'] = 21
    # pts = np.around(pts, decimals=1)
    # pts[:, 2] = 2
    annotation['keypoints'] = kpt
    annotation['category_id'] = 1
    # annotation['category_id'] = [annotation['category_id']]
    annotation['id'] = image['id']
    anno_list.append(annotation)

    count += 1

    if 1:
        # 画框
        cv2.rectangle(img, (int(x1 + 0.5), int(y1 + 0.5)), (int(x2 + 0.5), int(y2 + 0.5)), (0, 0, 255), 2)

        points = coord
        for i, p in enumerate(points):
            if i < 5:
                color = (255, 0, 0) # 大拇指 蓝色
            elif i >= 5 and i <= 8:
                color = (0, 0, 255) # 食指 红色
            elif i >= 9 and i <= 12:
                color = (0, 255, 0) # 中指 绿色
            elif i >= 13 and i <= 16:
                color = (255, 255, 0)
            elif i >= 17 and i <= 20:
                color = (0, 255, 255)
            cv2.circle(img, tuple(p), 3, color, 3)
            # cv2.putText(crop_img, str(i), p, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

            # cv2.imshow("1", img)
            # cv2.waitKey(0)

        # 画直线
        edges = [[0, 1], [1, 2], [2, 3], [3, 4],
                 [0, 5], [5, 6], [6, 7], [7, 8],
                 [0, 9], [9, 10], [10, 11], [11, 12],
                 [0, 13], [13, 14], [14, 15], [15, 16],
                 [0, 17], [17, 18], [18, 19], [19, 20]]
        for x, y in edges:
            if points[x][0] > 0 and points[x][1] > 0 and points[y][0] > 0 and points[y][1] > 0:
                cv2.line(img, tuple(points[x]), tuple(points[y]), (255, 0, 255), 2)

        # print(os.path.join(vis_save_path, im_s[-2] + "_right_" + im_s[-1]))
        # cv2.imwrite(os.path.join(vis_save_path, im_s[-2] + "_" + im_s[-3] + "_" + im_s[-4] + "_" + im_s[-1]), img)
        cv2.imshow("vis", img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
save_json_path = 'F:/1228_data_gesture/json_all.json'
categorie = categorie()
data_coco = data2coco(img_list, categorie, anno_list)
json.dump(data_coco, open(save_json_path, 'w'))#,indent=4