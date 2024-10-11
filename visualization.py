import cv2
import os
from utils import normalize
import numpy as np
import json

def visualizer(pathes, anomaly_map, img_size, save_path, cls_name):
    for idx, path in enumerate(pathes):
        
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        mask = normalize(anomaly_map[idx])
        vis = apply_ad_scoremap(vis, mask)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        save_vis = os.path.join(save_path, cls_name[idx], cls)
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        cv2.imwrite(os.path.join(save_vis, filename), vis)


# def visualizer(pathes, anomaly_map, img_size, save_path, cls_name, pr_sp):  # crop data 에 사용
#     print(f' \n pathes ===> {len(pathes)}, {pathes} \n')
#     # print(f' \n anomaly_map ===> {len(anomaly_map)}, {anomaly_map} \n')
#     # print(f' \n cls_name ===> {len(cls_name)}, {cls_name} \n')
#     # print(f' \n pr_sp ===> {len(pr_sp)}, {pr_sp} \n')
    
#     for idx, path in enumerate(pathes):
#         print(f'path ===> {path}')
#         #  path = C:/Users/dmkwo/Documents/soo/AnomalyCLIP/data/crop/cucumber/test/bad/V006_77_1_15_08_03_11_1_1096q_20201018_10.JPG
#         label_dir = path.replace('/test/', '/label/') + '.json'
#         try:
#             with open(label_dir, "r") as json_file:
#                     labels = json.load(json_file)

#             for point in labels['annotations']['points']:
#                 xtl = point['xtl']
#                 ytl = point['ytl']
#                 xbr = point['xbr']
#                 ybr = point['ybr']  

#         except (FileNotFoundError, json.JSONDecodeError) as e:
#                 print(f"Skipping file {label_dir} due to error: {e}")

#                 xtl = 0
#                 ytl = 0
#                 xbr = 0
#                 ybr = 0
#                 continue


#         class_name = path.split('/')[-4] # 오이, 가지, ..
#         cls = path.split('/')[-2] # good/bad
#         filename = path.split('/')[-1]
#         vis = cv2.rectangle(cv2.imread(path), (xtl, ytl), (xbr, ybr), (0, 0, 255), 50)
#         vis = cv2.cvtColor(cv2.resize(vis, (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB

#         mask = normalize(anomaly_map[idx])
#         vis = apply_ad_scoremap(vis, mask)

#         vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
#         # save_vis = os.path.join(save_path, f'abnormal_{cls_name[idx]}', cls)
#         save_vis = os.path.join(save_path, class_name, cls)
#         if not os.path.exists(save_vis):
#             os.makedirs(save_vis)
#         cv2.imwrite(os.path.join(save_vis, filename), vis)




def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
