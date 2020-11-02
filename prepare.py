#!/usr/bin/python3
#coding=utf-8

import os
import shutil
import numpy as np
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
import warnings
from glob import glob
import concurrent.futures

from preprocess import load_and_process, binarize_per_slice, all_slice_analysis, fill_hole, two_lung_only
from config_training import config

def resample(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')

def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM!=np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing, isflip

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 1.5 * np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask

def lumTrans(img):
    lungwin = np.array([-1200., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg*255).astype('uint8')
    return newimg

def lkds_create_mask(mhd_file):
    """this function is rewritten from load_and_process() from process1.py
    Add attributes of origin and isflip from mhd file
    """
    case_pixels, origin, spacing, isflip = load_itk_image(mhd_file)
    bw = binarize_per_slice(case_pixels, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)

    # remove the first few slices which have no lung/air areas.
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return case_pixels, bw1, bw2, spacing, origin, isflip

def savenpy_kaggle(id, annos, filelist, data_path, prep_folder):
    resolution = np.array([1, 1, 1])
    name = filelist[id]
    label = annos[annos[:,0] == name]

    # transpose to name, z, x(w), y(h), d
    label = label[:,[3,1,2,4]].astype('float')
    
    im, m1, m2, spacing = load_and_process(os.path.join(data_path, name))
    Mask = m1 + m2
    
    newshape = np.round(np.array(Mask.shape)*spacing/resolution)
    xx,yy,zz = np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
    extendbox = extendbox.astype('int')

    dm1 = process_mask(m1)
    dm2 = process_mask(m2)
    dilatedMask = dm1 + dm2
    extramask = dilatedMask ^ Mask

    bone_thresh = 210
    pad_value = 170
    im[np.isnan(im)] = -2000
    sliceim = lumTrans(im)
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim*extramask>bone_thresh
    sliceim[bones] = pad_value
    sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                        extendbox[1,0]:extendbox[1,1],
                        extendbox[2,0]:extendbox[2,1]]
    sliceim = sliceim2[np.newaxis,...]
    np.save(os.path.join(prep_folder,name+'_clean.npy'),sliceim)

    # Some stage1 cases have no nodule at all. Set label2 to [0,0,0,0]
    if len(label)==0:
        label2 = np.array([[0,0,0,0]])
    elif len(label[0])==0:
        label2 = np.array([[0,0,0,0]])
    elif label[0][0]==0:
        label2 = np.array([[0,0,0,0]])
    else:
        haslabel = 1  #Positive case with nodules.
        # Convert from voxel coordinate to world coordinate (in new resolution = [1,1,1]mm)
        # [nodule_n, z, x, y, d] --> [nodule_n, z, y, x, d]
        label2 = np.copy(label).T
        label2[:3] = label2[:3][[0,2,1]]
        label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        label2[3] = label2[3]*spacing[1]/resolution[1]
        label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
        label2 = label2[:4].T
    np.save(os.path.join(prep_folder,name+'_label.npy'),label2)
    print('{}: savenpy_kaggle completed.'.format(name))

def preprocess_kaggle():
    warnings.filterwarnings("ignore")

    prep_folder = config['preprocess_result_path']
    data_path = config['kaggle_data_path']
    finished_flag = '.flag_preprocess_kaggle'
    
    if not os.path.exists(finished_flag):
        alllabelfiles = config['kaggle_annos_path']    # Merge multiple csv into one numpy array
        tmp = []
        for f in alllabelfiles:
            content = np.array(pandas.read_csv(f))
            content = content[content[:, 0] != np.nan]
            tmp.append(content[:, :5])
        alllabel = np.concatenate(tmp, 0)

        if not os.path.isdir(prep_folder):
            os.makedirs(prep_folder, 0o755)

        print('\nStart pre-processing Kaggle dataset')

        exist_files = {f.split('_clean.npy')[0] for f in os.listdir(prep_folder) if f.endswith('_clean.npy')}
        filelist = {f for f in os.listdir(data_path)}
        filelist = list(filelist - exist_files)

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(savenpy_kaggle, f, annos=alllabel, filelist=filelist, data_path=data_path,
                                       prep_folder=prep_folder):f for f in range(len(filelist))}
            for future in concurrent.futures.as_completed(futures):
                filename = filelist[futures[future]]
                try:
                    _ = future.result()
                except:
                    print('{} failed.'.format(filename))

        print('\nEnd prep-rocessing Kaggle dataset.')
        f = open(finished_flag,"w+")
        f.close()
    return

def savenpy_luna(id, annos, filelist, luna_segment, luna_data, savepath):
    """
    Note: Dr. Chen adds malignancy label, so the label becomes (z,y,x,d,malignancy), <- but I cancelled it !
    """
    islabel = True
    isClean = True
    resolution = np.array([1, 1, 1])
    name = filelist[id]

    # Load mask, and calculate extendbox from the mask
    Mask, origin, spacing, isflip = load_itk_image(os.path.join(luna_segment, name+'.mhd'))
    if isflip:
        Mask = Mask[:,::-1,::-1]
    newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int')
    m1 = Mask==3
    m2 = Mask==4
    Mask = m1+m2
    
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T


    if isClean:
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1 + dm2
        Mask = m1 + m2
        extramask = dilatedMask ^ Mask  # '-' substration is deprecated in numpy, use '^'
        bone_thresh = 210
        pad_value = 170

        sliceim, origin, spacing, isflip = load_itk_image(os.path.join(luna_data, name+'.mhd'))
        if isflip:
            sliceim = sliceim[:,::-1,::-1]
            print('{}: flip!'.format(name))
        sliceim = lumTrans(sliceim)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = (sliceim*extramask)>bone_thresh
        sliceim[bones] = pad_value
        
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                            extendbox[1,0]:extendbox[1,1],
                            extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]

        np.save(os.path.join(savepath, name + '_clean.npy'),sliceim)

        np.save(os.path.join(savepath, name+'_spacing.npy'), spacing)
        np.save(os.path.join(savepath, name+'_extendbox.npy'), extendbox)
        np.save(os.path.join(savepath, name+'_origin.npy'), origin)
        np.save(os.path.join(savepath, name+'_mask.npy'), Mask)


    if islabel:
        this_annos = np.copy(annos[annos[:,0] == name])
        label = []

        if len(this_annos)>0:
            for c in this_annos:   # unit in mm  -->  voxel
                pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)  # (z,y,x)
                if isflip:
                    pos[1:] = Mask.shape[1:3] - pos[1:]   # flip in y and x coordinates
                d = c[4]/spacing[1]
                try:
                    malignancy = int(c[5])
                except IndexError:
                    malignancy = 0
                # label.append(np.concatenate([pos,[d],[malignancy]]))  # (z,y,x,d,malignancy)
                label.append(np.concatenate([pos,[d]]))  # (z,y,x,d)
            
        label = np.array(label)

        # Voxel --> resample to (1mm,1mm,1mm) voxel coordinate
        if len(label)==0:
            # label2 = np.array([[0,0,0,0,0]])
            label2 = np.array([[0,0,0,0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
            label2[3] = label2[3]*spacing[1]/resolution[1]
            label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
            # label2 = label2[:5].T   #(z,y,x,d,malignancy)
            label2 = label2[:4].T   #(z,y,x,d)

        np.save(os.path.join(savepath, name+'_label.npy'), label2)
        
    print('{} is done.'.format(name))

def preprocess_luna():
    luna_segment = config['luna_segment']
    savepath = config['preprocess_result_path']
    luna_data = config['luna_data']
    luna_label = config['luna_label']
    finished_flag = '.flag_preprocess_luna'

    print('starting preprocessing luna')

    # if not os.path.exists(finished_flag):
    if True:
        exist_files = {f.split('_clean.npy')[0] for f in os.listdir(savepath) if f.endswith('_clean.npy')}
        filelist = {f.split('.mhd')[0] for f in os.listdir(luna_data) if f.endswith('.mhd')}
        filelist = list(filelist - exist_files)
        annos = np.array(pandas.read_csv(luna_label))

        if not os.path.isdir(savepath):
            os.mkdir(savepath)

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(savenpy_luna, f, annos=annos, filelist=filelist,
                                       luna_segment=luna_segment, luna_data=luna_data, savepath=savepath):f for f in range(len(filelist))}
            for future in concurrent.futures.as_completed(futures):
                filename = filelist[futures[future]]
                try:
                    _ = future.result()
                except:
                    print('{} failed.'.format(filename))


    print('end preprocessing luna')
    f = open(finished_flag,"w+")
    f.close()
    return
    
def prepare_luna():
    """ Copy all subset of LUNA16 to one folder"""
    luna_root = config['luna_root']
    luna_allset = config['luna_data']
    finished_flag = '.flag_prepare_luna'

    if not os.path.exists(finished_flag):
        if not os.path.exists(luna_allset):
            os.makedirs(luna_allset)

        subsetdirs = [os.path.join(luna_root, 'subset{}'.format(i)) for i in range(10)]

        for d in subsetdirs:
            files = glob(os.path.join(d, '*'))
            for f in files:
                shutil.copy(f, luna_allset)
                print(os.path.join(luna_allset, os.path.basename(f)))

    f = open(finished_flag, "w+")
    f.close()
    return

def prepare_lkds():
    """
    Copy all subsets into allset. You have to combine train and val annotations.csv manually.
    """
    lkds_train = config['lkds_train']
    lkds_val = config['lkds_val']
    lkds_data = config['lkds_data']
    finished_flag = '.flag_prepare_lkds'

    if not os.path.exists(finished_flag):
        subsetdirs_train = [os.path.join(lkds_train, f) for f in os.listdir(lkds_train) if
                      f.startswith('train_subset') and os.path.isdir(os.path.join(lkds_train, f))]
        subsetdirs_val = [os.path.join(lkds_val, f) for f in os.listdir(lkds_val) if
                            f.startswith('val_subset') and os.path.isdir(os.path.join(lkds_val, f))]

        if not os.path.isdir(lkds_data):
            os.makedirs(lkds_data, 0o755)

        print('\nStart moving LKDS train subsets.')
        for d in subsetdirs_train:
            files = os.listdir(d)
            files.sort()
            for f in files:
                shutil.move(os.path.join(d, f), os.path.join(lkds_data, f))
                print(os.path.join(lkds_data, f))
        print('\nEnd moving LKDS train subsets.')

        print('\nStart moving LKDS val subsets.')
        for d in subsetdirs_val:
            files = os.listdir(d)
            files.sort()
            for f in files:
                shutil.move(os.path.join(d, f), os.path.join(lkds_data, f))
                print(os.path.join(lkds_data, f))
        print('\nEnd moving LKDS val subsets.')

    f = open(finished_flag, "w+")
    f.close()
    return

def savenpy_lkds(index, annos, filelist, lkds_data, savepath):
    isLabel = True
    isClean = True
    isInfo = False
    resolution = np.array([1, 1, 1])
    name = filelist[index]

    sliceim, m1, m2, spacing, origin, isflip = lkds_create_mask(os.path.join(lkds_data, name + '.mhd'))
    sliceim[np.isnan(sliceim)] = -2000
    sliceim = lumTrans(sliceim)

    if isflip:
        m1 = m1[:, ::-1, ::-1]
        m2 = m2[:, ::-1, ::-1]
        sliceim = sliceim[:, ::-1, ::-1]
        print('{} is flipped!'.format(name))

    Mask = m1 + m2
    newshape = np.round(np.array(Mask.shape) * spacing / resolution).astype('int')

    xx, yy, zz = np.where(Mask)
    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0, 0, 0], box[:, 0] - margin], 0),
                           np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]).T

    if isClean:
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1 + dm2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype('uint8')

        # fill the bone pixels with pad_value
        bones = (sliceim * extramask) > bone_thresh
        sliceim[bones] = pad_value
        sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
        # crop outside of mask, ie. extendbox
        sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1],
                            extendbox[1, 0]:extendbox[1, 1],
                            extendbox[2, 0]:extendbox[2, 1]]
        sliceim = sliceim2[np.newaxis, ...]
        np.save(os.path.join(savepath, name + '_clean.npy'), sliceim)

    if isLabel:
        this_annos = np.copy(annos[annos[:, 0] == name])
        label = []
        if len(this_annos) > 0:
            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)
                if isflip:
                    pos[1:] = Mask.shape[1:3] - pos[1:]
                label.append(np.concatenate([pos, [c[4] / spacing[1]]]))

        label = np.array(label)
        if len(label) == 0:
            label2 = np.array([[0, 0, 0, 0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
            label2[3] = label2[3] * spacing[1] / resolution[1]
            label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)
            label2 = label2[:4].T
        np.save(os.path.join(savepath, name + '_label.npy'), label2)

    if isInfo:
        preprocess_info = {'original_shape': sliceim.shape,
                           'original_spacing': spacing,
                           'resolution': resolution,
                           'new_shape': newshape,
                           'extendbox': extendbox,
                           'is_flip': isflip}
        np.save(os.path.join(savepath, name + '_info.npy'), preprocess_info)

    print('{} is done.'.format(name))
    return

def preprocess_lkds():
    warnings.filterwarnings("ignore")
    savepath = config['preprocess_result_path']
    lkds_data = config['lkds_data']
    lkds_label = config['lkds_label']
    finished_flag = '.flag_preprocess_lkds'
    blacklist = ['LKDS-00065', 'LKDS-00150', 'LKDS-00192', 'LKDS-00238', 'LKDS-00319', 'LKDS-00353',
                 'LKDS-00359', 'LKDS-00379', 'LKDS-00504', 'LKDS-00541', 'LKDS-00598', 'LKDS-00648',
                 'LKDS-00684', 'LKDS-00829', 'LKDS-00926', 'LKDS-00931']

    if os.path.isdir(savepath):
        exist_id = [f.split('_clean.npy')[0] for f in os.listdir(savepath) if f.endswith('_clean.npy') and f.startswith('LKDS')]
    else:
        os.makedirs(savepath, 0o755)
        exist_id = []

    if exist_id:
        print('These ids have been preprocessed:\n')
        for i in exist_id:
            print(i)
    else:
        print('No preprocessed id exists.\n')

    if not os.path.exists(finished_flag):
        filelist = [f.split('.mhd')[0] for f in os.listdir(lkds_data) if f.endswith('.mhd')]

        if exist_id:
            filelist = [f for f in filelist if f not in exist_id]

        if blacklist:
            filelist = [f for f in filelist if f not in blacklist]

        filelist.sort()
        annos = np.array(pandas.read_csv(lkds_label))

        print('start preprocessing LKDS\n')

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(savenpy_lkds, f, annos=annos, filelist=filelist, lkds_data=lkds_data, savepath=savepath):f for f in range(len(filelist))}
            for future in concurrent.futures.as_completed(futures):
                filename = filelist[futures[future]]
                try:
                    _ = future.result()
                except:
                    print('{} failed.'.format(filename))

        print('end preprocessing LKDS\n')
        f = open(finished_flag, "w+")
        f.close()
    return


if __name__=='__main__':
    # Pre-process Kaggle stage 1 DICOM files
    # preprocess_kaggle()

    # Copy all subsets of LUNA16 into one folder
    # prepare_luna()

    # Pre-process LUNA16 MHD files
    preprocess_luna()

    # Pre-process LKDS MHD files
    # preprocess_lkds()
