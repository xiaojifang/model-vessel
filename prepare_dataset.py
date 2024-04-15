import h5py
import numpy as np
import os.path
from PIL import Image
from glob import glob
from skimage.transform import resize
import cv2

raw_training_x_path_DRIVE = '/root/autodl-fs/DRIVE/training/images/*.tif'
raw_training_y_path_DRIVE = '/root/autodl-fs/DRIVE/training/1st_manual/*.gif'
raw_test_x_path_DRIVE = '/root/autodl-fs/DRIVE/test/images/*.tif'
raw_test_y_path_DRIVE = '/root/autodl-fs/DRIVE/test/1st_manual/*.gif'
raw_test_mask_path_DRIVE = '/root/autodl-fs/DRIVE/test/mask/*.gif'

raw_training_x_path_CHASEDB1 = '/root/autodl-fs/CHASEDB1/training/images/*.jpg'
raw_training_y_path_CHASEDB1 = '/root/autodl-fs/CHASEDB1/training/1st_manual/*1stHO.png'
raw_test_x_path_CHASEDB1 = '/root/autodl-fs/CHASEDB1/test/images/*.jpg'
raw_test_y_path_CHASEDB1 = '/root/autodl-fs/CHASEDB1/test/1st_manual/*1stHO.png'
raw_test_mask_path_CHASEDB1 = '/root/autodl-fs/CHASEDB1/test/mask/*mask.png'

raw_training_x_path_STARE = '/root/autodl-fs/STARE/training/stare-image/*.ppm'
raw_training_y_path_STARE = '/root/autodl-fs/STARE/training/labels-ah/*.ppm'
# raw_test_x_path_STARE = '/root/autodl-fs/STARE/test/stare-image/*.ppm'
# raw_test_y_path_STARE = '/root/autodl-fs/STARE/test/labels-ah/*.ppm'
raw_test_mask_path_STARE = '/root/autodl-fs/STARE/test/mask/*mask.png'
# raw_all_x_path_STARE = '/root/autodl-fs/STARE/training/image/*.ppm'
# raw_all_y_path_STARE = '/root/autodl-fs/STARE/training/labels-ah/*.ppm'
raw_test_x_path_STARE = '/root/autodl-fs/STARE/test/image/*.ppm'
raw_test_y_path_STARE = '/root/autodl-fs/STARE/test/labels-ah/*.ppm'

raw_training_x_path_CHUAC = '/root/autodl-fs/CHUAC/training/image/*.png'
raw_training_y_path_CHUAC = '/root/autodl-fs/CHUAC/training/mask/*.png'
raw_test_x_path_CHUAC = '/root/autodl-fs/CHUAC/test/image/*.png'
raw_test_y_path_CHUAC = '/root/autodl-fs/CHUAC/test/mask/*.png'
raw_test_mask_path_CHUAC = ''

raw_data_path = None
raw_data_path_DRIVE = [raw_training_x_path_DRIVE, raw_training_y_path_DRIVE, raw_test_x_path_DRIVE,
                       raw_test_y_path_DRIVE, raw_test_mask_path_DRIVE]
raw_data_path_CHASEDB1 = [raw_training_x_path_CHASEDB1, raw_training_y_path_CHASEDB1, raw_test_x_path_CHASEDB1,
                          raw_test_y_path_CHASEDB1, raw_test_mask_path_CHASEDB1]
# raw_data_path_STARE = [raw_all_x_path_STARE, raw_all_y_path_STARE]
raw_data_path_STARE = [raw_training_x_path_STARE, raw_training_y_path_STARE, raw_test_x_path_STARE,
                       raw_test_y_path_STARE, raw_test_mask_path_STARE]
raw_data_path_CHUAC = [raw_training_x_path_CHUAC, raw_training_y_path_CHUAC, raw_test_x_path_CHUAC,
                          raw_test_y_path_CHUAC, raw_test_mask_path_CHUAC]

HDF5_data_path = './data/HDF5/'
# 为每个数据集指定了期望的数据形状
DESIRED_DATA_SHAPE_DRIVE = (576, 576)
DESIRED_DATA_SHAPE_CHASEDB1 = (960, 960)
DESIRED_DATA_SHAPE_STARE = (592, 592)
DESIRED_DATA_SHAPE_CHUAC = (512, 512)
DESIRED_DATA_SHAPE = None
def get_radius(image_path):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 使用高斯模糊降低噪声
    img = cv2.medianBlur(img,5)
    # 使用霍夫变换检测圆
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    # 获取检测到的第一个圆的半径
    if circles is not None:
        circles = np.uint16(np.around(circles))
        _, _, radius = circles[0][0]
        return radius
    else:
        return None

#
# def generate_fov_mask(image_path, output_path, threshold=30):
#     # 读取图像
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     # 设置颜色阈值
#     _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
#
#     # 应用形态学操作进行清理
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#
#     # 获取图像的文件名（不包括扩展名）
#     filename = os.path.splitext(os.path.basename(image_path))[0]
#
#     # 为掩膜指定一个唯一的文件名
#     mask_filename = f"{filename}_mask.png"
#
#     # 保存掩膜
#     cv2.imwrite(os.path.join(output_path, mask_filename), mask)
def generate_fov_mask(image_path, output_path, threshold=40):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 设置颜色阈值
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    # 获取图像的文件名（不包括扩展名）
    filename = os.path.splitext(os.path.basename(image_path))[0]
    # 为掩膜指定一个唯一的文件名
    mask_filename = f"{filename}_mask.png"
    # 保存掩膜
    cv2.imwrite(os.path.join(output_path, mask_filename), mask)
# #
# def generate_fov_mask(image_path, output_path):
#     # 读取图像
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     # 获取图像的高度和宽度
#     height, width = img.shape
#     # 创建一个全黑的图像，大小和原图像相同
#     mask = np.zeros((height, width), np.uint8)
#     # 计算中心点的坐标
#     center = (width // 2, height // 2)
#     # 计算半径
#     radius = get_radius(image_path)
#     # 在掩膜上画一个白色的圆
#     cv2.circle(mask, center, radius, (255), thickness=-1)
#     # 获取图像的文件名（不包括扩展名）
#     filename = os.path.splitext(os.path.basename(image_path))[0]
#     # 为掩膜指定一个唯一的文件名
#     mask_filename = f"{filename}_mask.png"
#     # 保存掩膜
#     cv2.imwrite(os.path.join(output_path, mask_filename), mask)

#检查指定的原始数据路径是否已经存在 HDF5 文件。
def isHDF5exists(raw_data_path, HDF5_data_path):
    for raw in raw_data_path:
        if not raw: # raw 为 /root/autodl-fs/DRIVE/training/images/*.tif
            continue

        raw_splited = raw.split('/') # raw_splited 为 ['', 'root', 'autodl-fs', 'DRIVE', 'training', 'images', '*.tif']
        HDF5 = ''.join([HDF5_data_path, '/'.join(raw_splited[2:-1]), '/*.hdf5']) # 为./data/HDF5/autodl-fs/DRIVE/training/images/*.hdf5

        if len(glob(HDF5)) == 0:
            return False

    return True

# 从给定路径读取输入图像数据。处理掩膜和特定数据集的特殊情况。
def read_input(path):
    if path.find('mask') > 0 and (path.find('CHASEDB1') > 0 or path.find('STARE') > 0):
        fn = lambda x: 1.0 if x > 0.5 else 0
        x = np.array(Image.open(path).convert('L').point(fn, mode='1')) / 1.
    elif path.find('2nd') > 0 and path.find('DRIVE') > 0:
        x = np.array(Image.open(path)) / 1.
    elif path.find('_manual') > 0 and path.find('CHASEDB1') > 0:
        x = np.array(Image.open(path)) / 1.
    else:
        x = np.array(Image.open(path)) / 255.
    if x.shape[-1] == 3:
        return x
    else:
        return x[..., np.newaxis]

# 通过将图像调整大小到期望的形状或应用特定的转换来预处理数据。将图像转换为 NumPy 数组。
def preprocessData(data_path, dataset):
    global DESIRED_DATA_SHAPE

    data_path = list(sorted(glob(data_path)))
    if data_path[0].find('mask') > 0:
        return np.array([read_input(image_path) for image_path in data_path])
    else:
        return np.array([resize(read_input(image_path), DESIRED_DATA_SHAPE) for image_path in data_path])


def createHDF5(data, HDF5_data_path):
    try:
        os.makedirs(HDF5_data_path, exist_ok=True)
    except:
        pass
    f = h5py.File(HDF5_data_path + 'data.hdf5', 'w')
    f.create_dataset('data', data=data)
    return

# 将上述函数组合在一起，通过预处理原始数据并创建 HDF5 文件来准备数据集。
def prepareDataset(dataset):
    global raw_data_path, HDF5_data_path, raw_data_path_DRIVE, raw_data_path_CHASEDB1, raw_data_path_STARE, raw_data_path_CHUAC
    global DESIRED_DATA_SHAPE
    if dataset == 'DRIVE':
        DESIRED_DATA_SHAPE = DESIRED_DATA_SHAPE_DRIVE
    elif dataset == 'CHASEDB1':
        DESIRED_DATA_SHAPE = DESIRED_DATA_SHAPE_CHASEDB1
    elif dataset == 'STARE':
        DESIRED_DATA_SHAPE = DESIRED_DATA_SHAPE_STARE
    elif dataset == 'CHUAC':
        DESIRED_DATA_SHAPE = DESIRED_DATA_SHAPE_CHUAC

    if dataset == 'DRIVE':
        raw_data_path = raw_data_path_DRIVE
    elif dataset == 'CHASEDB1':
        raw_data_path = raw_data_path_CHASEDB1
    elif dataset == 'STARE':
        raw_data_path = raw_data_path_STARE
    elif dataset == 'CHUAC':
        raw_data_path = raw_data_path_CHUAC
    if isHDF5exists(raw_data_path, HDF5_data_path):
        return
    if dataset == 'CHASEDB1':
        for image_path in glob(raw_data_path[2]):
            generate_fov_mask(image_path, '/root/autodl-fs/CHASEDB1/test/mask')
    elif dataset == 'STARE':
        for image_path in glob(raw_data_path[2]):
            generate_fov_mask(image_path, '/root/autodl-fs/STARE/test/mask')
    for raw in raw_data_path:
        if not raw:
            continue

        raw_splited = raw.split('/')
        HDF5 = ''.join([HDF5_data_path, '/'.join(raw_splited[2:-1]), '/'])

        preprocessed = preprocessData(raw, dataset)
        createHDF5(preprocessed, HDF5)


def getTrainingData(XorY, dataset):
    global HDF5_data_path, raw_data_path_DRIVE, raw_data_path_CHASEDB1, raw_data_path_STARE

    if dataset == 'DRIVE':
        raw_training_x_path, raw_training_y_path = raw_data_path_DRIVE[:2]
    elif dataset == 'CHASEDB1':
        raw_training_x_path, raw_training_y_path = raw_data_path_CHASEDB1[:2]
    elif dataset == 'STARE':
        raw_training_x_path, raw_training_y_path = raw_data_path_STARE[:2]
    elif dataset == 'CHUAC':
        raw_training_x_path, raw_training_y_path = raw_data_path_CHUAC[:2]

    if XorY == 0:
        raw_splited = raw_training_x_path.split('/')
    else:
        raw_splited = raw_training_y_path.split('/')

    # data_path = ''.join([HDF5_data_path, dataset, '/', '/'.join(raw_splited[3:-1]), '/data.hdf5'])
    data_path = ''.join([HDF5_data_path, 'autodl-fs/', '/'.join(raw_splited[3:-1]),'/data.hdf5'])

    f = h5py.File(data_path, 'r')
    data = f['data']

    return data


def getTestData(XorYorMask, dataset):
    global HDF5_data_path, raw_data_path_DRIVE, raw_data_path_CHASEDB1, raw_data_path_STARE

    if dataset == 'DRIVE':
        raw_test_x_path, raw_test_y_path, raw_test_mask_path = raw_data_path_DRIVE[2:]
    elif dataset == 'CHASEDB1':
        raw_test_x_path, raw_test_y_path, raw_test_mask_path = raw_data_path_CHASEDB1[2:]
    elif dataset == 'STARE':
        raw_test_x_path, raw_test_y_path, raw_test_mask_path = raw_data_path_STARE[2:]
    elif dataset == 'CHUAC':
        raw_test_x_path, raw_test_y_path, raw_test_mask_path = raw_data_path_CHUAC[2:]

    if XorYorMask == 0:
        raw_splited = raw_test_x_path.split('/')
    elif XorYorMask == 1:
        raw_splited = raw_test_y_path.split('/')
    else:
        if not raw_test_mask_path:
            return None
        raw_splited = raw_test_mask_path.split('/')

    # data_path = ''.join([HDF5_data_path, dataset, '/', '/'.join(raw_splited[3:-1]), '/data.hdf5'])
    data_path = ''.join([HDF5_data_path, 'autodl-fs/', '/'.join(raw_splited[3:-1]),'/data.hdf5'])
    f = h5py.File(data_path, 'r')
    data = f['data']

    return data


def getallData(XorY, dataset):
    global HDF5_data_path, raw_data_path_STARE


    if dataset == 'STARE':
        raw_all_x_path, raw_all_y_path = raw_data_path_STARE[:2]


    if XorY == 0:
        raw_splited = raw_all_x_path.split('/')
    else:
        raw_splited = raw_all_y_path.split('/')

    # data_path = ''.join([HDF5_data_path, dataset, '/', '/'.join(raw_splited[3:-1]), '/data.hdf5'])
    data_path = ''.join([HDF5_data_path, 'autodl-fs/', '/'.join(raw_splited[3:-1]),'/data.hdf5'])

    f = h5py.File(data_path, 'r')
    data = f['data']

    return data