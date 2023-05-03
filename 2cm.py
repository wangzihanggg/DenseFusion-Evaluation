import _init_paths
import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import pandas as pd
from lib.knn.__init__ import KNearestNeighbor

knn = KNearestNeighbor(1)

def gettestdata(itemid,meta,cld,refine):
    num_pt_mesh_small = 500
    num_pt_mesh_large = 2600

    obj = meta['cls_indexes'].flatten().astype(np.int32)
    idx = list(obj).index(itemid)
    target_r = meta['poses'][:, :, idx][:, 0:3]
    target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
    #add_t = np.array([random.uniform(-noise_trans, noise_trans) for i in range(3)])

    dellist = [j for j in range(0, len(cld[obj[idx]]))]
    if refine:
        dellist = random.sample(dellist, len(cld[obj[idx]]) - num_pt_mesh_large)
    else:
        dellist = random.sample(dellist, len(cld[obj[idx]]) - num_pt_mesh_small)
    model_points = np.delete(cld[obj[idx]], dellist, axis=0)

    target = np.dot(model_points, target_r.T)
    ### 在test和eval模式时，add_noise为False
    """
    if add_noise:
        target = np.add(target, target_t + add_t)
    else:
        target = np.add(target, target_t)
    """
    target = np.add(target, target_t)
    
    return torch.from_numpy(target.astype(np.float32)), \
           torch.from_numpy(model_points.astype(np.float32)), \
           torch.LongTensor([int(obj[idx]) - 1])

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--epoch', type=str, default = 'current',  help='train epoch')
opt = parser.parse_args()

dataset_config_dir = 'datasets/ycb/dataset_config'
ycb_toolbox_dir = 'YCB_Video_toolbox'
result_wo_refine_dir = 'experiments/eval_result/ycb/original/Densefusion_wo_refine_result'
result_refine_dir = 'experiments/eval_result/ycb/original/Densefusion_iterative_result'
output_result_dir = 'experiments/eval_result/ycb/original/eval_accuracy'
accuracyresult = pd.DataFrame(columns = ['obj_num','accuracy','success','numcount'])

testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()
print(len(testlist))

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]

    input_file = open('{0}/models/{1}/points.xyz'.format(opt.dataset_root, class_input))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        #print(cld[class_id])
    input_file.close()
    cld[class_id] = np.array(cld[class_id])
    class_id += 1
print(cld.keys())

objlist = [i+1 for i in range(21)]

success_count = [0 for i in range(21)]
num_count = [0 for i in range(21)]
for i in range(len(testlist)):
    meta = scio.loadmat('{0}/{1}-meta.mat'.format(opt.dataset_root, testlist[i]))
    meta_lst = meta['cls_indexes'].flatten().astype(np.int32)
    posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % i))
    label = np.array(posecnn_meta['labels'])
    posecnn_rois = np.array(posecnn_meta['rois'])
    posecnn_lst = posecnn_rois[:, 1:2].flatten()
    lst = [i for i in posecnn_lst if i in meta_lst]
    #read_wo_refine_pose = scio.loadmat('{0}/{1}_{2}.mat'.format(result_wo_refine_dir, '%04d' % now, opt.epoch))
    read_refine_pose = scio.loadmat('{0}/{1}_{2}.mat'.format(result_refine_dir, '%04d' % i, opt.epoch))
    
    for idx in range(len(lst)):  #每张里面有多个物体，以此循环获取
        itemid = lst[idx]
        # with refine：我这里只计算有refine的，想要计算没有refine的，就把最后一个参数改成True，r和t读取对应的文件就行
        target, model_points, obj_idx = gettestdata(itemid,meta,cld,True)
        target, model_points, obj_idx = Variable(target).cuda(),\
                                    Variable(model_points).cuda(), \
                                    Variable(obj_idx).cuda()
        #获取存储的r和t
        pred_r = read_refine_pose['poses'][idx,:4]
        pred_t = read_refine_pose['poses'][idx,4:]
        model_points = model_points.cpu().detach().numpy()
        my_r = quaternion_matrix(pred_r)[:3, :3] 
        my_t = pred_t
        pred = np.dot(model_points, my_r.T) + my_t
        target = target.cpu().detach().numpy()
        #计算ADD-S
        pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
        if dis < 0.02:
            success_count[int(itemid)-1] += 1
            print('No.{0} Obj {1} Pass! Distance: {2}'.format(i, itemid, dis))
            #fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
        else:
            print('No.{0} Obj {1} NOT Pass! Distance: {2}'.format(i, itemid, dis))
            #fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
        num_count[int(itemid)-1] += 1
for i in range(21):
    print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
    accuracyresult = accuracyresult.append({'obj_num': objlist[i], 'accuracy': float(success_count[i]) / num_count[i],'success':success_count[i],'numcount':num_count[i]}, ignore_index=True)
    #fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
accuracyresult = accuracyresult.append({'obj_num': 'ALL', 'accuracy': float(sum(success_count)) / sum(num_count),'success':sum(success_count),'numcount':sum(num_count)}, ignore_index=True)
accuracyresult.to_csv('{0}/eval_result_accuracy_{1}.csv'.format(output_result_dir,opt.epoch),index=0,encoding="gbk")