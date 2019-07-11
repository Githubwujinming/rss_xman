import os
from collections import defaultdict
import config.rssia_config as cfg
name_img = defaultdict(list)

def getProjection(sub_path,dir ):
    child_list = os.listdir(sub_path)
    i = 0
    child_list.sort()
    for sub_dir in child_list:
        child_path = os.path.join(dir, sub_dir)
        name_img[i].append(child_path)
        i += 1

def main(base_dir):
    output_dir = base_dir+"/val.txt"
    if os.path.exists(output_dir):
        os.remove(output_dir)
    dir_list = os.listdir(base_dir)
    dir_list.sort()
    for sub_dir in dir_list:
        dir = sub_dir
        sub_path = os.path.join(base_dir, sub_dir)
        getProjection(sub_path,dir)
    f = open(output_dir, 'w+')
    for key in name_img.keys():
        img_info = name_img[key]
        for info in img_info:
            f.write(info)
            f.write(' ')
        f.write('\n')
    f.close()
if __name__=="__main__":
    base_dir = "/home/ubuntu/PycharmProjects/datasets/rssai_tif_croped/val"
    # base_dir = "/home/ubuntu/PycharmProjects/datasets/rssrai2019_new/val"
    main(base_dir)