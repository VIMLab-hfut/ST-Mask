import os
import shutil

#图片文件夹
file_dir_input = "E:/picture_cfx/train/classify_12_21complicate"
#json文件夹
json_dir_input = "E:/picture_cfx/车缝线-总结/深度学习/整理/ALL_pic_json"
#导出文件夹
json_dir_output = "E:/picture_cfx/train/classify_12_21complicate"

pic_form = 'jpg'
bz_form = 'json'

index = 0
index_json = 0
pic_name = []
json_name = []
files = os.listdir(file_dir_input)
for file in files:
    name, ext = file.split('.', 1)
    if ext == pic_form:
        index += 1
        pic_name.append(name)


os.chdir(json_dir_input)
for dirpath, subdirs, files_json in os.walk(os.getcwd()):
    for file_json in files_json:
        name, ext = file_json.split('.', 1)
        if ext == bz_form:
            if name in pic_name:
                shutil.copy(file_json, json_dir_output)
                index_json += 1
                print('已复制{}到指定文件夹'.format(file_json))

print('图片总数量为{}'.format(index))
print('json总数量为{}'.format(index_json))
