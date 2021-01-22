import shutil
import os

json_path = 'F:/1228_data_gesture\json'
image_path = 'F:/1228_data_gesture\crop_iamges2'

json_list = []
for j in os.listdir(json_path):
    json_list.append(j.split('.')[0])

for im in os.listdir(image_path):
    if im.split('.')[0] in json_list:
        continue
    else:
        src = os.path.join(image_path, im)
        dst = os.path.join("F:/1228_data_gesture/3", im)
        shutil.copyfile(src, dst)