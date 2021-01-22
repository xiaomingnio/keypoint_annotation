from tkinter import *
from PIL.ImageTk import PhotoImage
from tkinter.filedialog import askdirectory
import numpy as np
import json
import random
import mediapipe as mp
import cv2
from PIL import Image, ImageTk
import os
import glob


file_id = 0
file_list = []
save_path = ''

def save(p):
    # save_path = "./save"
    data = {}
    print(file_list[file_id])
    data['image_file'] = file_list[file_id].split("\\")[-1]
    json_name = data['image_file'].split(".")[0] + ".json"
    data['keypoints'] = (np.array(list(p.values()))/800).tolist()
    # print(data)
    json_str = json.dumps(data)
    with open(os.path.join(save_path, json_name), 'w') as json_file:
        json_file.write(json_str)


def vis(p):
    kps = list(p.values())
    img = cv2.imread(file_list[file_id])
    img = cv2.resize(img, (800, 800))
    keypoints = np.array(kps)
    xs = keypoints[:, 0]
    ys = keypoints[:, 1]
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
        cv2.circle(img, p, 4, color, -1)
        cv2.putText(img, str(i), p, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    # 画直线
    edges = [[0, 1], [1, 2], [2, 3], [3, 4],
             [0, 5], [5, 6], [6, 7], [7, 8],
             [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16],
             [0, 17], [17, 18], [18, 19], [19, 20]]
    for x, y in edges:
        if points[x][0] > 0 and points[x][1] > 0 and points[y][0] > 0 and points[y][1] > 0:
            cv2.line(img, points[x], points[y], (0, 0, 255), 2)
    cv2.imshow("show_vis", img)
    # cv2.waitKey(0)


def next_img():
    global file_id, idx_list, lines_idx_list
    idx_list = []
    lines_idx_list = []
    file_id += 1
    w.delete(ALL)
    gen(file_id)

def get_image_dir():
    global file_list
    imageDir = askdirectory()
    file_list = glob.glob(imageDir + "/*.jpg")

def get_save_dir():
    global save_path
    save_path = askdirectory()

class DragPoint:
    def __init__(self, w, idx_list,lines_idx_list,file_id, coord):
        self.moving = False
        self.mx = 0
        self.my = 0
        self.idx_list = idx_list # [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62]
        self.w = w
        self.locx = 0
        self.locy = 0
        self.idx = -1
        self.file_id = file_id
        self.coord = coord
        self.lines_idx_list = lines_idx_list
        self.edges = [[0, 1], [1, 2], [2, 3], [3, 4],
                 [0, 5], [5, 6], [6, 7], [7, 8],
                 [0, 9], [9, 10], [10, 11], [11, 12],
                 [0, 13], [13, 14], [14, 15], [15, 16],
                 [0, 17], [17, 18], [18, 19], [19, 20]]


    def moveimg(self, event):
        print("-----------------move-----------------")
        print("self.idx_list: ", self.idx_list)
        print("self.lines_idx_list: ", self.lines_idx_list)
        # print(event.x, event.y, self.mx, self.my)
        if self.moving and abs(event.x - self.mx) <= 30 and abs(event.y - self.my) <= 30:
            self.w.move(self.idx, event.x - self.mx, event.y - self.my)
            self.mx, self.my = event.x, event.y

        for i, line in enumerate(self.edges):
            # 找到 self.idx在self.idx_list中的索引
            if int((self.idx%84)/3) == line[0]:
                # print(line[1]*3+2)
                self.w.coords(self.lines_idx_list[i], event.x, event.y, self.coord[line[1]*3+2+84*self.file_id][0], self.coord[line[1]*3+2+84*self.file_id][1])
            elif int((self.idx%84)/3) == line[1]:
                # print("line_id: ", lines_idx_list[i])
                self.w.coords(self.lines_idx_list[i], self.coord[line[0]*3+2+84*self.file_id][0], self.coord[line[0]*3+2+84*self.file_id][1], event.x, event.y)

    def ButtonRelease(self, event):
        # print("self.idx_list: ", self.idx_list)
        self.mx = self.my = 0
        self.moving = False
        # print(self.idx_list)
        if self.idx != -1:
            component = event.widget
            co = component.coords(self.idx)
            self.locx = (co[0] + co[2]) / 2
            self.locy = (co[1] + co[3]) / 2
            self.coord[self.idx] = [self.locx, self.locy]
            # print("locx, locy:", self.locx,",", self.locy)

    def ButtonPush(self, event):
        component = event.widget
        # 计算所有组件坐标与当前鼠标位置的距离
        min_dist = 9999999
        min_dist_idx = -1
        for idx in self.idx_list:
            co = component.coords(idx)
            # 得到圆形中心点坐标
            locx = (co[0] + co[2])/2
            locy = (co[1] + co[3])/2
            # print("locx, locy:", locx,",", locy)
            # print("ButtonPush: ", mx, my)
            # 计算欧式距离
            dist = np.sqrt((locx - event.x)**2 + (locy - event.y)**2)
            if dist <= min_dist:
                min_dist = dist
                min_dist_idx = idx
                self.locx = locx
                self.locy = locy

        if abs(self.locx - event.x) <= 30 and abs(self.locy- event.y) <= 30:
            self.moving = True
            self.mx, self.my = event.x, event.y
            self.idx = min_dist_idx




root = Tk()
w = Canvas(root, width=900, height=850)
w.grid(row=0, column=0, rowspan=40, sticky=W+N+S+E)
idx_list = []
lines_idx_list = []
coord = {}
p = DragPoint(w, idx_list,lines_idx_list,file_id, coord)
Button(root, text="start", command=lambda:gen(file_id)).grid(row=0, column=1, sticky=NW)
Button(root, text="next", command=next_img).grid(row=1, column=1, sticky=NW)
Button(root, text="save", command=lambda:save(p.coord)).grid(row=2, column=1, sticky=NW)
Button(root, text="vis", command=lambda:vis(p.coord)).grid(row=3, column=1, sticky=NW)
Button(root, text="选择图片目录", command=get_image_dir).grid(row=4, column=1, sticky=NW)
Button(root, text="选择保存目录", command=get_save_dir).grid(row=5, column=1, sticky=NW)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5)
edges = [[0, 1], [1, 2], [2, 3], [3, 4],
             [0, 5], [5, 6], [6, 7], [7, 8],
             [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16],
             [0, 17], [17, 18], [18, 19], [19, 20]]


def gen(file_id):
    global pil_image, tkimg, idx_list, lines_idx_list
    idx_list.clear()
    lines_idx_list.clear()
    file_name = file_list[file_id]
    pil_image = Image.open(file_name)
    # 缩放到指定大小
    img_w = 800
    img_h = 800
    pil_image = pil_image.resize((img_w, img_h), Image.ANTIALIAS)
    tkimg = ImageTk.PhotoImage(pil_image)
    w.create_image(0, 0, image=tkimg, anchor='nw')
    image = cv2.imread(file_name)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print("results: ", results.multi_hand_landmarks)
    point_nums = 21
    init_coords = []
    if results.multi_hand_landmarks is not None:
        hand_landmarks = results.multi_hand_landmarks[0]

        # 预训练打标坐标
        # 创建21个点
        for i in range(point_nums):
            init_coords.append((hand_landmarks.landmark[i].x * img_w, hand_landmarks.landmark[i].y * img_h))
    else:
        for i in range(point_nums):
            init_coords.append((100, 100+20*i))

    color_list = np.load("color.npy")
    r = 5
    coord = {}
    for i in range(point_nums):
        c_x = init_coords[i][0]
        c_y = init_coords[i][1]
        draw_x = 850
        draw_y = 180 + 30*i
        point_id = w.create_oval(c_x-r, c_y-r, c_x + r, c_y + r, width=0, fill=color_list[i][0])
        idx_list.append(point_id)
        print("point_id", point_id)
        w.create_oval(draw_x-2*r, draw_y-2*r, draw_x + 2*r, draw_y + 2*r, width=0, fill=color_list[i][0])
        w.create_text(draw_x+25, draw_y, text=str(i))
        coord[point_id] = [c_x, c_y]
    p.coord = coord

    # 画直线
    for x, y in edges:
        line_id = w.create_line(init_coords[x][0], init_coords[x][1], init_coords[y][0], init_coords[y][1])
        lines_idx_list.append(line_id)
        print("line: ", line_id)
    p.idx_list = idx_list
    p.lines_idx_list = lines_idx_list
    p.file_id = file_id
    w.bind('<Button-1>', p.ButtonPush)
    w.bind('<B1-Motion>', p.moveimg)
    w.bind('<ButtonRelease-1>', p.ButtonRelease)

    print("gen: ", idx_list)
    print(coord)

# 创建保存按钮
mainloop()