import cv2
import json
import numpy as np
import os

json_path = "F:/1228_data_gesture/json_all"

for json_f in os.listdir(json_path):
    with open(os.path.join(json_path, json_f), 'r') as f:
        ann = json.load(f)

    print(ann['image_file'])

    img = cv2.imread("F:/1228_data_gesture/images_all/" + ann['image_file'])

    h, w = img.shape[:2]
    keypoints = np.array(ann['keypoints'])
    xs = keypoints[:, 0]*w
    ys = keypoints[:, 1]*h
    points = [(int(xs[i] + 0.5), int(ys[i] + 0.5)) for i in range(len(xs))]
    for i, p in enumerate(points):
        if i < 5:
            color = (255, 0, 0)
        elif i >= 5 and i <= 8:
            color = (0, 0, 255)
        elif i >= 9 and i <= 12:
            color = (0, 255, 0)
        elif i >= 13 and i <= 16:
            color = (255, 255, 0)
        elif i >= 17 and i <= 20:
            color = (0, 255, 255)
        cv2.circle(img, p, 2, color, -1)
        cv2.putText(img, str(i), p, cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

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
            cv2.line(img, points[x], points[y], (0, 0, 255), 1)

    img = cv2.resize(img, (800, 800))
    cv2.imshow("img", img)
    cv2.waitKey(100)

