import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataset import shanghaitech_hr_skip
import joblib
import json
import cv2
import copy
###############################################################################################################################################
##################################### Part 1: combine human with other classes for state machine ##############################################
###############################################################################################################################################

human_dir = '/root/Downloads/STG-NF/data/ShanghaiTech/pose/'
other_cls_dir = '/root/Downloads/STG-NF_other_cls_objs/data/ShanghaiTech/pose/test_other_cls/'
other_cls_list = ['1', '2', '3', '7']

gt = joblib.load(os.path.join(human_dir, 'test_gt.pkl'))
human_scores = joblib.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/test_scores_30_pred_4.pkl') # (os.path.join(human_dir, 'test_scores.pkl'))

normalize_inside_test_or_from_train = 1
proportion_in_training = 0.99 # many training instances have higher anomaly scores than test ones
normalize_or_not = 1
merge_motion_with_appearance = 1
multi_scale_fusion = 1

def score_auc(scores_np, gt):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    ############## 魔改 加点 vecolity信息 ###### 0 1 是反的
    # # np.save('utils\\scores_nfpose.npy', scores_np)
    # vel = np.load('utils\\final_scores_velocity.npy')
    # final_score =  scores_np - vel
    # auc = roc_auc_score(gt, final_score)
    auc = roc_auc_score(gt, scores_np)
    return auc

def smooth_scores(scores_arr, sigma=7):
    for s in scores_arr:
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr

for other_cls in other_cls_list:
    other_cls_scores = joblib.load(os.path.join(other_cls_dir, other_cls + '_scores.pkl'))
    for video_name in other_cls_scores:
        # valid segments
        curr_cls_curr_video = json.load(open(os.path.join(other_cls_dir, other_cls, video_name + '_alphapose_tracked_person.json'), 'r'))
        valid_frame_id_list = []
        for curr_cls_curr_video_key in curr_cls_curr_video:
            valid_frame_id_list += [int(x) for x in list(curr_cls_curr_video[curr_cls_curr_video_key].keys())]
        valid_frame_id_list = np.unique(valid_frame_id_list).tolist()

        # minus average and divided by variance
        # if abs(np.min(human_scores[video_name]) - np.max(human_scores[video_name])) != 0 and abs(np.min(other_cls_scores[video_name]) - np.max(other_cls_scores[video_name])) != 0:
        #     human_scores[video_name] = (human_scores[video_name] - np.max(human_scores[video_name])) / abs(np.min(human_scores[video_name]) - np.max(human_scores[video_name]))
        #     other_cls_scores[video_name] = (other_cls_scores[video_name] - np.max(other_cls_scores[video_name])) / abs(np.min(other_cls_scores[video_name]) - np.max(other_cls_scores[video_name]))
        #     human_scores[video_name] = human_scores[video_name] + other_cls_scores[video_name]

        for x in range(len(human_scores[video_name].tolist())):
            if x in valid_frame_id_list:
                human_scores[video_name][x] = min([human_scores[video_name].tolist()[x], other_cls_scores[video_name].tolist()[x]])

# normalize
if normalize_or_not:
    if normalize_inside_test_or_from_train == 0:
        human_scores_max_value, human_scores_min_value = np.max([max(x) for x in [human_scores[x] for x in human_scores]]), \
                                                         np.min([min(x) for x in [human_scores[x] for x in human_scores]])
    else:
        train_scores_part1 = np.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/train_scores_30_pred_4_part1.npy', allow_pickle=True)
        train_scores_part2 = np.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/train_scores_30_pred_4_part2.npy', allow_pickle=True)
        train_scores_part3 = np.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/train_scores_30_pred_4_part3.npy', allow_pickle=True)
        train_scores_part4 = np.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/train_scores_30_pred_4_part4.npy', allow_pickle=True)
        human_scores_max_value = np.sort(np.concatenate((train_scores_part1, train_scores_part2, train_scores_part3, train_scores_part4), axis=0))[-1]
        human_scores_min_value = np.sort(np.concatenate((train_scores_part1, train_scores_part2, train_scores_part3, train_scores_part4), axis=0))[int(len(np.concatenate((train_scores_part1, train_scores_part2, train_scores_part3, train_scores_part4), axis=0)) * (1.0 - proportion_in_training))]

    for video_name in human_scores:
        human_scores[video_name] = (human_scores[video_name] - human_scores_max_value) / abs(human_scores_min_value - human_scores_max_value)

###############################################################################################################################################
if multi_scale_fusion == 1:
    human_10_pred_2_dir = '/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/'
    human_scores_10_pred_2 = joblib.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/test_scores_30_pred_6.pkl') # joblib.load(os.path.join(human_10_pred_2_dir, 'test_scores_10_pred_2.pkl'))
    # normalize
    if normalize_or_not:
        if normalize_inside_test_or_from_train == 0:
            human_scores_max_value, human_scores_min_value = np.max([max(x) for x in [human_scores_10_pred_2[x] for x in human_scores_10_pred_2]]), \
                                                             np.min([min(x) for x in [human_scores_10_pred_2[x] for x in human_scores_10_pred_2]])
        else:
            train_scores_part1 = np.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/train_scores_30_pred_6_part1.npy', allow_pickle=True)
            train_scores_part2 = np.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/train_scores_30_pred_6_part2.npy', allow_pickle=True)
            train_scores_part3 = np.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/train_scores_30_pred_6_part3.npy', allow_pickle=True)
            train_scores_part4 = np.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/train_scores_30_pred_6_part4.npy', allow_pickle=True)
            human_scores_max_value = np.sort(np.concatenate((train_scores_part1, train_scores_part2, train_scores_part3, train_scores_part4), axis=0))[-1]
            human_scores_min_value = np.sort(np.concatenate((train_scores_part1, train_scores_part2, train_scores_part3, train_scores_part4), axis=0))[int(len(np.concatenate((train_scores_part1, train_scores_part2, train_scores_part3, train_scores_part4), axis=0)) * (1.0 - proportion_in_training))]

        for video_name in human_scores_10_pred_2:
            human_scores_10_pred_2[video_name] = (human_scores_10_pred_2[video_name] - human_scores_max_value) / abs(human_scores_min_value - human_scores_max_value)

    for video_name in human_scores:
        for time_idx in range(len(human_scores[video_name])):
            human_scores[video_name][time_idx] = np.min([human_scores[video_name][time_idx], human_scores_10_pred_2[video_name][time_idx]])

    # human_10_pred_2_dir = '/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/'
    # human_scores_10_pred_2 = joblib.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/test_scores_30_pred_4_8626.pkl') # joblib.load(os.path.join(human_10_pred_2_dir, 'test_scores_10_pred_2.pkl'))
    # # normalize
    # if normalize_or_not:
    #     if normalize_inside_test_or_from_train == 0:
    #         human_scores_max_value, human_scores_min_value = np.max([max(x) for x in [human_scores_10_pred_2[x] for x in human_scores_10_pred_2]]), \
    #                                                          np.min([min(x) for x in [human_scores_10_pred_2[x] for x in human_scores_10_pred_2]])
    #     else:
    #         train_scores_part1 = np.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/train_scores_30_pred_4_part1.npy', allow_pickle=True)
    #         train_scores_part2 = np.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/train_scores_30_pred_4_part2.npy', allow_pickle=True)
    #         train_scores_part3 = np.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/train_scores_30_pred_4_part3.npy', allow_pickle=True)
    #         train_scores_part4 = np.load('/root/Downloads/STG-NF-multiscale/data/ShanghaiTech/pose/train_scores_30_pred_4_part4.npy', allow_pickle=True)
    #         human_scores_max_value = np.sort(np.concatenate((train_scores_part1, train_scores_part2, train_scores_part3, train_scores_part4), axis=0))[-1]
    #         human_scores_min_value = np.sort(np.concatenate((train_scores_part1, train_scores_part2, train_scores_part3, train_scores_part4), axis=0))[int(len(np.concatenate((train_scores_part1, train_scores_part2, train_scores_part3, train_scores_part4), axis=0)) * (1.0 - proportion_in_training))]
    #
    #     for video_name in human_scores_10_pred_2:
    #         human_scores_10_pred_2[video_name] = (human_scores_10_pred_2[video_name] - human_scores_max_value) / abs(human_scores_min_value - human_scores_max_value)
    #
    # for video_name in human_scores:
    #     for time_idx in range(len(human_scores[video_name])):
    #         human_scores[video_name][time_idx] = np.min([human_scores[video_name][time_idx], human_scores_10_pred_2[video_name][time_idx]])
###############################################################################################################################################
##################################### Part 2: introduce appearances ###########################################################################
###############################################################################################################################################
img_folder = '/root/Downloads/STG-NF/data/ShanghaiTech/images/frames_part/'
test_separate_video_json = np.load('/root/Downloads/STG-NF/data/ShanghaiTech/pose/test_other_cls/test_clip_lengths.npy', allow_pickle=True)
test_deep_features_array = np.load('/root/Downloads/Accurate-Interpretable-VAD-master/test_deep_features_scores.npy')
human_scores_appearance = {}

videoes_lens = {}
for json_name in [x for x in os.listdir(img_folder)]:
    if '.avi' in json_name:
        cap = cv2.VideoCapture(os.path.join(img_folder, json_name))
        frames_num = cap.get(7)
        videoes_lens[json_name.split('.avi')[0]] = int(frames_num)
    else:
        videoes_lens[json_name] = len(os.listdir(os.path.join(img_folder, json_name)))

sorted_keys = sorted([x for x in videoes_lens.keys() if len(x) == 6], key = lambda x:int(x.split('_')[0] + x.split('_')[1])) + sorted([x for x in videoes_lens.keys() if len(x) == 7], key = lambda x:int(x.split('_')[0] + x.split('_')[1]))
sorted_dict = dict(zip(sorted_keys, [videoes_lens[key] for key in sorted_keys]))
videoes_lens = sorted_dict

frame_cnt = 0
for test_video_len_idx in range(len(test_separate_video_json)):
    # print('test ' + str(test_video_len_idx))
    test_video_len = test_separate_video_json[test_video_len_idx]
    # maybe two videos have same length
    curr_video_name = [x for x in videoes_lens if len(x) == 7][test_video_len_idx]  # [x for x in videoes_lens if videoes_lens[x] == int(train_video_len - frame_cnt)][0]
    assert (videoes_lens[curr_video_name] == (test_video_len - frame_cnt))
    human_scores_appearance[curr_video_name] = test_deep_features_array[frame_cnt: test_video_len]
    # not "+=", must be "="
    frame_cnt = test_video_len

# normalize
if normalize_or_not:
    if normalize_inside_test_or_from_train == 0:
        human_scores_appearance_max_value, human_scores_appearance_min_value = np.max([max(x) for x in [human_scores_appearance[x] for x in human_scores_appearance]]), \
                                                                               np.min([min(x) for x in [human_scores_appearance[x] for x in human_scores_appearance]])
    else:
        human_scores_appearance_max_value, human_scores_appearance_min_value = np.load('/root/Downloads/STG-NF/data/ShanghaiTech/pose/train_appearance_scores.npy', allow_pickle=True)[int(1154722 * proportion_in_training)], \
                                                                               np.load('/root/Downloads/STG-NF/data/ShanghaiTech/pose/train_appearance_scores.npy', allow_pickle=True)[0]# 46.34192, 0.0

    for video_name in human_scores_appearance:
        human_scores_appearance[video_name] = -(human_scores_appearance[video_name] - human_scores_appearance_min_value) / abs(human_scores_appearance_max_value - human_scores_appearance_min_value)

gt_np = np.concatenate([gt[x] for x in gt])  # 40791帧
human_scores_appearance_np = np.concatenate([human_scores_appearance[x] for x in human_scores_appearance])  # 40791
auc = score_auc(human_scores_appearance_np, gt_np)
print('Only appearance: ' + str(auc))

# video-level auc
video_level_appearance_auc = {}
for video_key in gt:
    if np.min(gt[video_key]) == np.max(gt[video_key]):
        video_level_appearance_auc[video_key] = None
        continue
    video_level_appearance_auc[video_key] = score_auc(human_scores_appearance[video_key], gt[video_key])
###############################################################################################################################################
##################################### Part 3: compute scores ##################################################################################
###############################################################################################################################################
# for video_name in human_scores:
#     # valid segments
#     other_cls_list = [1, 2, 3, 7] # , 8, 10, 13, 24, 25, 26, 28, 29, 30, 34, 35, 36, 37, 38, 67]
#     for other_cls in other_cls_list:
#         if not os.path.exists(os.path.join(other_cls_dir, str(other_cls), video_name + '_alphapose_tracked_person.json')):
#             continue
#         curr_cls_curr_video = json.load(open(os.path.join(other_cls_dir, str(other_cls), video_name + '_alphapose_tracked_person.json'), 'r'))
#         valid_frame_id_list = []
#         for curr_cls_curr_video_key in curr_cls_curr_video:
#             valid_frame_id_list += [int(x) for x in list(curr_cls_curr_video[curr_cls_curr_video_key].keys())]
#         valid_frame_id_list = np.unique(valid_frame_id_list).tolist()
#         for time_idx in valid_frame_id_list:
#             human_scores[video_name][time_idx] = min([human_scores[video_name][time_idx], human_scores_appearance[video_name][time_idx]])

# for video_name in ['10_0042', '01_0054', '01_0136', '03_0039']: #, '07_0008', '08_0077',
#     for time_idx in range(len(human_scores[video_name])):
#         human_scores[video_name][time_idx] = min([human_scores[video_name][time_idx], human_scores_appearance[video_name][time_idx]])
    ######################## human_scores[video_name] = human_scores_appearance[video_name]

############ dict(zip(human_scores_appearance.keys(), [np.min(human_scores_appearance[key]) for key in human_scores_appearance]))
# for video_name in ['10_0042', '01_0054', '01_0136', '03_0039', '07_0008', '08_0077']: #, '07_0008', '08_0077',
#     for time_idx in range(len(human_scores[video_name])):
#         human_scores[video_name][time_idx] = np.min([human_scores[video_name][time_idx], human_scores_appearance[video_name][time_idx]])
# '01_0051', '01_0054', '01_0064', '01_0131', '01_0135', '01_0162', '03_0031', '03_0039', '03_0041', '03_0060', '05_0017', '05_0018', '05_0019', '05_0020', '05_0021', '05_0023', '05_0024', '07_0008', '07_0047', '09_0057', '11_0176'


# max_anomaly_score_each_video = dict(zip(human_scores.keys(), [np.min(human_scores[key]) for key in human_scores]))
# for video_name in [x for x in human_scores if max_anomaly_score_each_video[x] >= -1.0]:
#     other_cls_list = [1, 2, 3, 7]# , 8, 10, 13, 24, 25, 26, 28, 29, 30, 34, 35, 36, 37, 38, 67]
#     for other_cls in other_cls_list:
#         if not os.path.exists(os.path.join(other_cls_dir, str(other_cls), video_name + '_alphapose_tracked_person.json')):
#             continue
#         curr_cls_curr_video = json.load(open(os.path.join(other_cls_dir, str(other_cls), video_name + '_alphapose_tracked_person.json'), 'r'))
#         valid_frame_id_list = []
#         for curr_cls_curr_video_key in curr_cls_curr_video:
#             valid_frame_id_list += [int(x) for x in list(curr_cls_curr_video[curr_cls_curr_video_key].keys())]
#         valid_frame_id_list = np.unique(valid_frame_id_list).tolist()
#         for time_idx in valid_frame_id_list:#range(len(human_scores[video_name])):
#             human_scores[video_name][time_idx] = np.min([float(human_scores[video_name][time_idx]), float(human_scores_appearance[video_name][time_idx])])

#################################################################################################################################################################
    # other_cls_list = [1, 2, 3, 7]#, 8, 10, 13, 24, 25, 26, 28, 29, 30, 34, 35, 36, 37, 38, 67]
    # valid_frame_id_list = []
    # for other_cls in other_cls_list:
    #     if not os.path.exists(os.path.join(other_cls_dir, str(other_cls), video_name + '_alphapose_tracked_person.json')):
    #         continue
    #     curr_cls_curr_video = json.load(open(os.path.join(other_cls_dir, str(other_cls), video_name + '_alphapose_tracked_person.json'), 'r'))
    #     for curr_cls_curr_video_key in curr_cls_curr_video:
    #         valid_frame_id_list += [int(x) for x in list(curr_cls_curr_video[curr_cls_curr_video_key].keys())]
    # valid_frame_id_list = np.unique(valid_frame_id_list).tolist()
#################################################################################################################################################################

if merge_motion_with_appearance == 1:
    human_scores_backup = copy.deepcopy(human_scores)
    train_scores_part1 = np.load('/root/Downloads/STG-NF/data/ShanghaiTech/pose/train_scores_part1.npy', allow_pickle=True)
    train_scores_part2 = np.load('/root/Downloads/STG-NF/data/ShanghaiTech/pose/train_scores_part2.npy', allow_pickle=True)
    train_scores_part3 = np.load('/root/Downloads/STG-NF/data/ShanghaiTech/pose/train_scores_part3.npy', allow_pickle=True)
    train_scores_part4 = np.load('/root/Downloads/STG-NF/data/ShanghaiTech/pose/train_scores_part4.npy', allow_pickle=True)
    train_pose_scores = np.sort(np.concatenate((train_scores_part1, train_scores_part2, train_scores_part3, train_scores_part4), axis=0)) # np.sort(np.concatenate([human_scores[x] for x in human_scores])) #         # [int(862946 * (proportion_in_training))]
    test_pose_scores = np.sort(np.concatenate([human_scores_backup[x] for x in human_scores_backup]))
    test_frames_probability_of_appearance = json.load(open('/root/Downloads/STG-NF/data/ShanghaiTech/pose/test_frames_probability_of_appearance.json', 'r'))
    for video_name in human_scores:
        other_cls_list = [1, 2, 3, 7]#, 8, 10, 13, 24, 25, 26, 28, 29, 30, 34, 35, 36, 37, 38, 67]
        whether_this_video_is_tracked = 0
        for other_cls in other_cls_list:
            if os.path.exists(os.path.join(other_cls_dir, str(other_cls), video_name + '_alphapose_tracked_person.json')):
                whether_this_video_is_tracked = 1
        if (whether_this_video_is_tracked == 1) or (video_name not in test_frames_probability_of_appearance):
            continue
        print(video_name)
        for time_idx in range(len(human_scores[video_name])):
            human_scores[video_name][time_idx] = np.min([float(human_scores[video_name][time_idx]), test_pose_scores[int(test_frames_probability_of_appearance[video_name][time_idx] * (len(test_pose_scores) - 1))] ])
            # human_scores[video_name][time_idx] = np.min([float(human_scores[video_name][time_idx]), human_scores_appearance[video_name][time_idx] ])
            # human_scores[video_name][time_idx] = np.max([human_scores[video_name][time_idx], np.min(human_scores_backup[video_name])])
        # human_scores = smooth_scores(human_scores)


gt_np = np.concatenate([gt[x] for x in gt])  # 40791帧
human_scores_np = np.concatenate([human_scores[x] for x in human_scores])  # 40791
auc = score_auc(human_scores_np, gt_np)

print(auc)

# video-level auc
video_level_auc = {}
for video_key in gt:
    if np.min(gt[video_key]) == np.max(gt[video_key]):
        video_level_auc[video_key] = None
        continue
    video_level_auc[video_key] = score_auc(human_scores[video_key], gt[video_key])
    # print(video_key + ': ' + str(video_level_auc[-1]))
# '04_0011' includes all normal behaviors
video_level_auc['04_0011'] = score_auc(np.concatenate((human_scores['04_0011'], human_scores['04_0012']), axis=0), \
                                       np.concatenate((gt['04_0011'], gt['04_0012']), axis=0))

print(np.mean([video_level_auc[x] for x in video_level_auc if video_level_auc[x] != None]))


