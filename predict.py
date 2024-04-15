
import os
import tensorflow as tf
# from keras.backend import tensorflow_backend
from tensorflow.keras import backend as K
import skimage.transform
import numpy as np
import tensorflow.keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# session = tf.Session(config=config)
# tensorflow_backend.set_session(session)

from utils import define_model, prepare_dataset, crop_prediction, unetpp
from utils.evaluate import evaluate
from keras.layers import ReLU
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
import cv2
from sklearn.metrics import roc_auc_score
from keras.models import load_model


import math
import pickle





def double_threshold_iteration(img, h_thresh, l_thresh, save=True):

    h, w = img.shape
    # img = np.array(1 / (1 + np.exp(-img)) * 255, dtype=np.uint8)  # 使用 sigmoid 函数
    img= img *255
    bin = np.where(img >= h_thresh * 255, 255, 0).astype(np.uint8)
    gbin = bin.copy()
    gbin_pre = gbin - 1
    while (gbin_pre.all() != gbin.all()):
        gbin_pre = gbin.copy()
        for i in range(1, h - 1):  # 修正循环范围，避免超出数组边界
            for j in range(1, w - 1):
                if gbin[i, j] == 0 and img[i, j] < h_thresh * 255 and img[i, j] >= l_thresh * 255:
                    neighbors = [gbin[i - 1, j - 1], gbin[i - 1, j], gbin[i - 1, j + 1],
                                 gbin[i, j - 1], gbin[i, j + 1],
                                 gbin[i + 1, j - 1], gbin[i + 1, j], gbin[i + 1, j + 1]]
                    if sum(neighbors) >= 255 * 4:
                        gbin[i, j] = 255

    return gbin / 255
def predict(ACTIVATION='ReLU', dropout=0.1, batch_size=32, repeat=4, minimum_kernel=32,
            epochs=200, iteration=0, crop_size=128, stride_size=3, DATASET='DRIVE'):
    prepare_dataset.prepareDataset(DATASET)
    test_data = [prepare_dataset.getTestData(0, DATASET),
                 prepare_dataset.getTestData(1, DATASET),
                 prepare_dataset.getTestData(2, DATASET)]

    IMAGE_SIZE = None
    if DATASET == 'DRIVE':
        IMAGE_SIZE = (565, 584)
    elif DATASET == 'CHASEDB1':
        IMAGE_SIZE = (999, 960)
    elif DATASET == 'STARE':
        IMAGE_SIZE = (700, 605)

    gt_list_out = {}
    pred_list_out = {}
    for out_id in range(iteration + 1):
        try:
            os.makedirs(f"./output/{DATASET}/crop_size_{crop_size}/out{out_id + 1}/", exist_ok=True)
            gt_list_out.update({f"out{out_id + 1}": []})
            pred_list_out.update({f"out{out_id + 1}": []})

        except:
            pass

    activation = globals()[ACTIVATION]
    # model = define_model.get_unet(minimum_kernel=32, do=0, activation=ReLU, iteration=0)
    # model = define_model.get_Attention()
    model = unetpp.unetpp()
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    total_params = trainable_count + non_trainable_count
    print('Total params: {:,}'.format(total_params))
    # 如果你想将参数量转换为MB
    total_params_MB = total_params * 4 / (1024 ** 2)  # 一个参数通常是4字节（32位浮点数）
    print('Total size of parameters: {:,} MB'.format(total_params_MB))



    model_name = f"Final_Emer_Iteration_{iteration}_cropsize_{crop_size}_epochs_{epochs}"
    print("Model : %s" % model_name)
    # load_path = f"trained_model/{DATASET}/{model_name}.hdf5"
    load_path = f"/root/autodl-fs/IterNet/IterNet/trained_model/DRIVE/Final_crop_size_128_epochs_100/9.hdf5"

    # model = load_model(load_path)  # 加载模型
    # print("Model compile configuration: ", model.get_config())

    model.load_weights(load_path, by_name=False)

    imgs = test_data[0]
    segs = test_data[1]
    masks = test_data[2]

    # for i in tqdm(range(len(imgs))):
    for i in tqdm([6, 14]):
        img = imgs[i]
        seg = segs[i]
        if masks:
            mask = masks[i]
        print("i",i)
        patches_pred, new_height, new_width, adjustImg = crop_prediction.get_test_patches(img, crop_size, stride_size)
        preds = model.predict(patches_pred) #
        preds_ =[]
        preds_.append(preds)
        preds =[]
        preds =preds_.copy()
        out_id = 0
        for pred in preds:
            pred_patches = crop_prediction.pred_to_patches(pred, crop_size, stride_size)

            pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
            pred_imgs = pred_imgs[:, 0:prepare_dataset.DESIRED_DATA_SHAPE[0], 0:prepare_dataset.DESIRED_DATA_SHAPE[0],
                        :]
            probResult = pred_imgs[0, :, :, 0]
            pred_ = probResult
            with open(f"./output/{DATASET}/crop_size_{crop_size}/out{out_id + 1}/{i + 1:02}.pickle", 'wb') as handle:
                pickle.dump(pred_, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pred_ = resize(pred_, IMAGE_SIZE[::-1])
            # pred_ = double_threshold_iteration(pred_, 0.5,0.4)
            if masks:
                mask_ = mask
                mask_ = resize(mask_, IMAGE_SIZE[::-1])
            seg_ = seg
            seg_ = resize(seg_, IMAGE_SIZE[::-1])
            gt_ = (seg_ > 0.5).astype(int)
            gt_flat = []
            pred_flat = []
            for p in range(pred_.shape[0]):
                for q in range(pred_.shape[1]):
                    if not masks or mask_[p, q] > 0.5:  # Inside the mask pixels only
                        gt_flat.append(gt_[p, q])
                        pred_flat.append(pred_[p, q])

            gt_list_out[f"out{out_id + 1}"] += gt_flat
            pred_list_out[f"out{out_id + 1}"] += pred_flat

            pred_ = 255. * (pred_ - np.min(pred_)) / (np.max(pred_) - np.min(pred_))
            cv2.imwrite(f"./output/{DATASET}/crop_size_{crop_size}/out{out_id + 1}/{i + 1:02}.png", pred_)
            out_id += 1
    # for out_id in range(iteration + 1)[-1:]:
    #     print('\n\n', f"out{out_id + 1}")
    #     evaluate(gt_list_out[f"out{out_id + 1}"], pred_list_out[f"out{out_id + 1}"], DATASET)
    evaluate(gt_list_out[f"out{1}"], pred_list_out[f"out{1}"], pred_list_out[f"out{1}"],DATASET)

if __name__ == "__main__":
    predict(batch_size=16, epochs=100, stride_size=24, DATASET='DRIVE')
