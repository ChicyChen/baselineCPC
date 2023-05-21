import os
import csv
import glob
import pandas as pd
from tqdm import tqdm 
from joblib import delayed, Parallel
import random

def write_list(data_list, path, ):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for row in data_list:
            if row: writer.writerow(row)
    print('split saved to %s' % path)


def main_UCF101_full(f_root, splits_root, csv_root='data/ucf101/'):
    '''generate training/testing split, count number of available frames, save in csv'''
    # get action list
    action_dict = {}
    action_file = os.path.join(splits_root, 'classInd.txt')
    action_df = pd.read_csv(action_file, sep=' ', header=None)
    for _, row in action_df.iterrows():
        act_id, act_name = row
        act_id = int(act_id) - 1 # let id start from 0
        action_dict[act_name] = act_id


    if not os.path.exists(csv_root): os.makedirs(csv_root)

    all_set = []
    video_name_set = []

    for which_split in [1,2,3]:
        
        train_split_file = os.path.join(splits_root, 'trainlist%02d.txt' % which_split)
        with open(train_split_file, 'r') as f:
            for line in f:
                tmp_path = ''
                tmp_path = line.split(' ')[0][0:-4]
                if tmp_path in video_name_set:
                    pass
                else:
                    video_name_set.append(tmp_path)
                    action_name = tmp_path.split('/')[0]
                    action_id = action_dict[action_name]
                    vpath = os.path.join(f_root, tmp_path) + '/'
                    all_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg'))), action_id])

        test_split_file = os.path.join(splits_root, 'testlist%02d.txt' % which_split)
        with open(test_split_file, 'r') as f:
            for line in f:
                tmp_path = ''
                tmp_path = line.split(' ')[0][0:-4]
                if tmp_path in video_name_set:
                    pass
                else:
                    video_name_set.append(tmp_path)
                    action_name = tmp_path.split('/')[0]
                    action_id = action_dict[action_name]
                    vpath = os.path.join(f_root, tmp_path) + '/'
                    all_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg'))), action_id])

    for i in range(5):
        random.shuffle(all_set)
    
    total_num = len(all_set)
    train_num = int(total_num*0.7)
    train_set = all_set[:train_num]
    test_set = all_set[train_num:]

    write_list(train_set, os.path.join(csv_root, 'train.csv'))
    write_list(test_set, os.path.join(csv_root, 'test.csv'))
    write_list(all_set, os.path.join(csv_root, 'all.csv'))
        

def main_UCF101(f_root, splits_root, csv_root='data/ucf101/'):
    '''generate training/testing split, count number of available frames, save in csv'''
    # get action list
    action_dict = {}
    action_file = os.path.join(splits_root, 'classInd.txt')
    action_df = pd.read_csv(action_file, sep=' ', header=None)
    for _, row in action_df.iterrows():
        act_id, act_name = row
        act_id = int(act_id) - 1 # let id start from 0
        action_dict[act_name] = act_id

    if not os.path.exists(csv_root): os.makedirs(csv_root)
    for which_split in [1,2,3]:
        train_set = []
        test_set = []
        
        train_split_file = os.path.join(splits_root, 'trainlist%02d.txt' % which_split)
        with open(train_split_file, 'r') as f:
            for line in f:
                tmp_path = ''
                tmp_path = line.split(' ')[0][0:-4]

                action_name = tmp_path.split('/')[0]
                action_id = action_dict[action_name]

                vpath = os.path.join(f_root, tmp_path)

                train_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg'))), action_id])

        test_split_file = os.path.join(splits_root, 'testlist%02d.txt' % which_split)
        with open(test_split_file, 'r') as f:
            for line in f:
                tmp_path = ''
                tmp_path = line.split(' ')[0][0:-4]

                if tmp_path[-1] == '.':
                    tmp_path = tmp_path[:-1]
                    print(tmp_path)

                action_name = tmp_path.split('/')[0]
                action_id = action_dict[action_name]

                vpath = os.path.join(f_root, tmp_path)

                
                test_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg'))), action_id])

        write_list(train_set, os.path.join(csv_root, 'train_split%02d.csv' % which_split))
        write_list(test_set, os.path.join(csv_root, 'test_split%02d.csv' % which_split))


def main_HMDB51_full(f_root, splits_root, csv_root='data/hmdb51/'):
    '''generate training/testing split, count number of available frames, save in csv'''
    action_dict_encode = {}
    idx = 0

    if not os.path.exists(csv_root): os.makedirs(csv_root)

    train_set = []
    test_set = []
    all_set = []
    video_name_set = []

    for which_split in [1,2,3]:
        split_files = sorted(glob.glob(os.path.join(splits_root, '*_test_split%d.txt' % which_split)))
        assert len(split_files) == 51
        for split_file in split_files:
            action_name = os.path.basename(split_file)[0:-16]

            if idx < 51:
                action_dict_encode[action_name] = idx
                idx = idx + 1
            action_id = action_dict_encode[action_name]

            with open(split_file, 'r') as f:
                for line in f:
                    video_name = line.split(' ')[0]

                    if video_name in video_name_set:
                        pass
                    else:
                        video_name_set.append(video_name)
                        vpath = os.path.join(f_root, action_name, video_name[0:-4]) + '/'
                        all_set.append([vpath, len(glob.glob(glob.escape(vpath)+"/*.jpg")), action_id])
    
    for i in range(5):
        random.shuffle(all_set)
    
    total_num = len(all_set)
    train_num = int(total_num*0.7)
    train_set = all_set[:train_num]
    test_set = all_set[train_num:]

    write_list(train_set, os.path.join(csv_root, 'train.csv'))
    write_list(test_set, os.path.join(csv_root, 'test.csv'))
    write_list(all_set, os.path.join(csv_root, 'all.csv'))


def main_HMDB51(f_root, splits_root, csv_root='data/hmdb51/'):
    '''generate training/testing split, count number of available frames, save in csv'''

    # pathname = "/home/siyich/baselineCPC/HMDB51/frame/fencing/Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_f_cm_np2_le_goo_0"
    # listlist = glob.glob(glob.escape(pathname)+"/*.jpg")
    # print(len(listlist))

    action_dict_encode = {}
    idx = 0
    if not os.path.exists(csv_root): os.makedirs(csv_root)
    for which_split in [1,2,3]:
        train_set = []
        test_set = []
        split_files = sorted(glob.glob(os.path.join(splits_root, '*_test_split%d.txt' % which_split)))
        assert len(split_files) == 51
        for split_file in split_files:
            action_name = os.path.basename(split_file)[0:-16]

            if idx < 51:
                action_dict_encode[action_name] = idx
                idx = idx + 1
            action_id = action_dict_encode[action_name]

            with open(split_file, 'r') as f:
                for line in f:
                    video_name = line.split(' ')[0][0:-4]

                    _type = line.split(' ')[1]
                    vpath = os.path.join(f_root, action_name, video_name)

                    # if "Die_Another_Day_-_Fencing_Scene_Part_1_[HD]" in video_name:
                    #     print(vpath)
                    #     vlist = glob.glob(vpath)
                    #     print(len(vlist))

                    if _type == '1':
                        train_set.append([vpath, len(glob.glob(glob.escape(vpath)+"/*.jpg")), action_id])
                    elif _type == '2':
                        test_set.append([vpath, len(glob.glob(glob.escape(vpath)+"/*.jpg")), action_id])

        write_list(train_set, os.path.join(csv_root, 'train_split%02d.csv' % which_split))
        write_list(test_set, os.path.join(csv_root, 'test_split%02d.csv' % which_split))

"""
### For Kinetics ###
def get_split(root, split_path, mode):
    print('processing %s split ...' % mode)
    print('checking %s' % root)
    split_list = []
    split_content = pd.read_csv(split_path).iloc[:,0:4]
    split_list = Parallel(n_jobs=64)\
                 (delayed(check_exists)(row, root) \
                 for i, row in tqdm(split_content.iterrows(), total=len(split_content)))
    return split_list

def check_exists(row, root):
    dirname = '_'.join([row['youtube_id'], '%06d' % row['time_start'], '%06d' % row['time_end']])
    full_dirname = os.path.join(root, row['label'], dirname)
    if os.path.exists(full_dirname):
        n_frames = len(glob.glob(os.path.join(full_dirname, '*.jpg')))
        return [full_dirname, n_frames]
    else:
        return None

def main_Kinetics400(mode, k400_path, f_root, csv_root='data/kinetics400'):
    train_split_path = os.path.join(k400_path, 'kinetics_train/train.csv')
    val_split_path = os.path.join(k400_path, 'kinetics_val/validate.csv')
    test_split_path = os.path.join(k400_path, 'kinetics_test/test.csv')
    if not os.path.exists(csv_root): os.makedirs(csv_root)
    if mode == 'train':
        train_split = get_split(os.path.join(f_root, 'train_split'), train_split_path, 'train')
        write_list(train_split, os.path.join(csv_root, 'train_split.csv'))
    elif mode == 'val':
        val_split = get_split(os.path.join(f_root, 'val_split'), val_split_path, 'val')
        write_list(val_split, os.path.join(csv_root, 'val_split.csv'))
    elif mode == 'test':
        test_split = get_split(f_root, test_split_path, 'test')
        write_list(test_split, os.path.join(csv_root, 'test_split.csv'))
    else:
        raise IOError('wrong mode')
"""

if __name__ == '__main__':
    # f_root is the frame path
    # edit 'your_path' here: 

    main_UCF101(f_root='UCF101/frame', 
                splits_root='UCF101/splits_classification',
                csv_root='data/ucf101')

    # main_HMDB51(f_root='HMDB51/frame',
    #             splits_root='HMDB51/split/testTrainMulti_7030_splits',
    #             csv_root='data/hmdb51')

    # main_Kinetics400(mode='train', # train or val or test
    #                  k400_path='Kinetics',
    #                  f_root='Kinetics400/frame')

    main_UCF101_full(f_root='UCF101/frame', 
                splits_root='UCF101/splits_classification')

    # main_HMDB51_full(f_root='HMDB51/frame',
    #             splits_root='HMDB51/split/testTrainMulti_7030_splits')

    # main_HMDB51_full(f_root='HMDB51/frame',
    #             splits_root='HMDB51/split/testTrainMulti_7030_splits')

    main_UCF101(f_root='UCF101/frame_240', 
                splits_root='UCF101/splits_classification',
                csv_root='data/ucf101_240/')

    # main_HMDB51(f_root='HMDB51/frame_240',
    #             splits_root='HMDB51/split/testTrainMulti_7030_splits',
    #             csv_root='data/hmdb51_240/')

    main_UCF101_full(f_root='UCF101/frame_240', 
                splits_root='UCF101/splits_classification',
                csv_root='data/ucf101_240/')

    # main_HMDB51_full(f_root='HMDB51/frame_240',
    #             splits_root='HMDB51/split/testTrainMulti_7030_splits',
    #             csv_root='data/hmdb51_240/')
