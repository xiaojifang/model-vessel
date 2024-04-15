import numpy as np

# def get_test_patches_3D(img, crop_size, stride_size, rl=False):
#     test_img = []
#     test_img.append(img)
#     test_img = np.asarray(test_img)
    
#     #     test_img_adjust=img_process(test_img,rl=rl)
#     test_img_adjust = test_img[0]
    
#     test_imgs = paint_border_3D(test_img_adjust, crop_size, stride_size)

#     test_img_patch = extract_patches_3D(test_imgs, crop_size, stride_size)
    
#     return test_img_patch, test_imgs.shape[0], test_imgs.shape[1], test_imgs.shape[2], test_img_adjust

# def get_test_patches(img, crop_size, stride_size, rl=False):
#     print("img",img.shape)
#     test_img = []

#     test_img.append(img)
#     test_img = np.asarray(test_img)
#     print("test_img",test_img.shape)
#     #     test_img_adjust=img_process(test_img,rl=rl)
#     test_img_adjust = test_img
#     print(type(test_img_adjust))
#     test_imgs = paint_border(test_img_adjust, crop_size, stride_size)

#     test_img_patch = extract_patches(test_imgs, crop_size, stride_size)

#     return test_img_patch, test_imgs.shape[1], test_imgs.shape[2], test_img_adjust

# def extract_patches_3D(full_imgs, crop_size, stride_size):
#     patch_height = crop_size
#     patch_width = crop_size
#     patch_depth = 8  # 设置裁剪深度
#     stride_height = stride_size
#     stride_width = stride_size
#     stride_depth = 8
    
#     img_d = full_imgs.shape[0]  # 图像的深度
#     img_h = full_imgs.shape[1]  # 图像的高度
#     img_w = full_imgs.shape[2]  # 图像的宽度
#     # 这是整个图像上的补丁数量
#     N_patches_img = ((img_d - patch_depth) // stride_depth + 1) * (
#             (img_h - patch_height) // stride_height + 1) * ((img_w - patch_width) // stride_width + 1)
#     # N_patches_tot = N_patches_img * full_imgs.shape[0]
#     N_patches_tot = N_patches_img 
    
#     patches = np.empty((N_patches_tot, patch_depth, patch_height, patch_width, full_imgs.shape[3]))
    
#     iter_tot = 0  # 总补丁数
    
#     for d in range((img_d - patch_depth) // stride_depth + 1):
#         for h in range((img_h - patch_height) // stride_height + 1):
#             for w in range((img_w - patch_width) // stride_width + 1):
#                 patch = full_imgs[d * stride_depth:(d * stride_depth) + patch_depth,
#                         h * stride_height:(h * stride_height) + patch_height,
#                         w * stride_width:(w * stride_width) + patch_width, :]
                
#                 patches[iter_tot] = patch
#                 iter_tot += 1  # 总补丁数
#     assert (iter_tot == N_patches_tot)
#     return patches


# def extract_patches(full_imgs, crop_size, stride_size):
#     patch_height = crop_size
#     patch_width = crop_size
#     stride_height = stride_size
#     stride_width = stride_size

#     # assert (len(full_imgs.shape) == 4)  # 4D arrays
#     img_h = full_imgs.shape[1]  # height of the full image
#     img_w = full_imgs.shape[2]  # width of the full image

#     # assert ((img_h - patch_height) % stride_height == 0 and (img_w - patch_width) % stride_width == 0)
#     N_patches_img = ((img_h - patch_height) // stride_height + 1) * (
#             (img_w - patch_width) // stride_width + 1)  # // --> division between integers
#     N_patches_tot = N_patches_img * full_imgs.shape[0]

#     patches = np.empty((N_patches_tot, patch_height, patch_width, full_imgs.shape[3]))
#     iter_tot = 0  # iter over the total number of patches (N_patches)
#     for i in range(full_imgs.shape[0]):  # loop over the full images
#         for h in range((img_h - patch_height) // stride_height + 1):
#             for w in range((img_w - patch_width) // stride_width + 1):
#                 patch = full_imgs[i, h * stride_height:(h * stride_height) + patch_height,
#                         w * stride_width:(w * stride_width) + patch_width, :]
#                 patches[iter_tot] = patch
#                 iter_tot += 1  # total
#     assert (iter_tot == N_patches_tot)
#     return patches

# # 用于处理3D图像数据的边界填充函数。
# # 它的目的是通过将额外的空间填充到图像的边界来确保所有图像块都具有相同的大小，以便在后续处理中进行有效处理
# def paint_border_3D(imgs, crop_size, stride_size):
#     patch_height = crop_size  # 48
#     patch_width = crop_size
#     patch_depth = 8
#     stride_height = stride_size  # 24
#     stride_width = stride_size
#     stride_depth =8
    
#     # assert (len(imgs.shape) == 4)
#     img_d = imgs.shape[0]  # depth of the full image
#     img_h = imgs.shape[1]  # height of the full image
#     img_w = imgs.shape[2]  # width of the full image
    
#     leftover_d = (img_d - patch_depth) % stride_depth
#     leftover_h = (img_h - patch_height) % stride_height  # leftover on the h dim
#     leftover_w = (img_w - patch_width) % stride_width  # leftover on the w dim
#     full_imgs = None
#     if (leftover_d != 0):  # change dimension of img_d
#         tmp_imgs = np.zeros((img_d + (stride_depth - leftover_d), imgs.shape[1], img_w, imgs.shape[3]))
#         tmp_imgs[0:img_d, 0:imgs.shape[1], 0:img_w, 0:imgs.shape[3]] = imgs
#         full_imgs = tmp_imgs

#     if (leftover_h != 0):  # change dimension of img_h
#         tmp_imgs = np.zeros((imgs.shape[0], img_h + (stride_height - leftover_h), img_w, imgs.shape[3]))
#         tmp_imgs[0:imgs.shape[0], 0:img_h, 0:img_w, 0:imgs.shape[3]] = imgs
#         full_imgs = tmp_imgs

#     if (leftover_w != 0):  # change dimension of img_w
#         depth = 8
#         tmp_imgs = np.zeros((imgs.shape[0], depth, imgs.shape[1], img_w + (stride_width - leftover_w), imgs.shape[3]))
        
#         for i in range(depth):
#             tmp_imgs = imgs[:, :, :, :, 0:1]
#         full_imgs = tmp_imgs

#     return full_imgs if full_imgs is not None else imgs









# def paint_border(imgs, crop_size, stride_size):
#     patch_height = crop_size
#     patch_width = crop_size
#     stride_height = stride_size
#     stride_width = stride_size
    
#     # assert (len(imgs.shape) == 4)
#     img_h = imgs.shape[1]  # height of the full image
#     img_w = imgs.shape[2]  # width of the full image
#     leftover_h = (img_h - patch_height) % stride_height  # leftover on the h dim
#     leftover_w = (img_w - patch_width) % stride_width  # leftover on the w dim
#     full_imgs = None
#     if (leftover_h != 0):  # change dimension of img_h
#         tmp_imgs = np.zeros((imgs.shape[0], img_h + (stride_height - leftover_h), img_w, imgs.shape[3]))
#         tmp_imgs[0:imgs.shape[0], 0:img_h, 0:img_w, 0:imgs.shape[3]] = imgs
#         full_imgs = tmp_imgs
#     if (leftover_w != 0):  # change dimension of img_w
#         tmp_imgs = np.zeros(
#             (full_imgs.shape[0], full_imgs.shape[1], img_w + (stride_width - leftover_w), full_imgs.shape[3]))
#         tmp_imgs[0:imgs.shape[0], 0:imgs.shape[1], 0:img_w, 0:full_imgs.shape[3]] = imgs
#         full_imgs = tmp_imgs
#         #     print("new full images shape: \n" +str(full_imgs.shape))
#         return full_imgs
    
#     else:
#         return imgs


# def pred_to_patches(pred, crop_size, stride_size):
#     return pred
#     patch_height = crop_size
#     patch_width = crop_size

#     seg_num = 0
#     #     print(pred.shape)

#     assert (len(pred.shape) == 3)  # 3D array: (Npatches,height*width,2)

#     pred_images = np.empty((pred.shape[0], pred.shape[1], seg_num + 1))  # (Npatches,height*width)
#     pred_images[:, :, 0:seg_num + 1] = pred[:, :, 0:seg_num + 1]
#     pred_images = np.reshape(pred_images, (pred_images.shape[0], patch_height, patch_width, seg_num + 1))
#     return pred_images

# def pred_to_patches_3D(pred, crop_size, stride_size):
#     depth = 8
#     patch_depth = min(crop_size, depth)
#     patch_height = crop_size
#     patch_width = crop_size
    
#     seg_num = 0

#     assert (len(pred.shape) == 5)  # 4D array: (Npatches, depth*height*width, 2)

#     pred_volumes = np.empty((pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3], seg_num + 1))  # (Npatches, depth*height*width)
#     pred_volumes[:, :, :, :, 0:seg_num + 1] = pred[:, :, :, :, 0:seg_num + 1]
#     pred_volumes = np.reshape(pred_volumes, (pred_volumes.shape[0], patch_depth, patch_height, patch_width, seg_num + 1))
#     return pred_volumes




# def recompone_overlap_3D(preds, crop_size, stride_size, img_d, img_h, img_w):
    
#     assert (len(preds.shape) == 5)  # 5D arrays

#     patch_d = 8
#     patch_h = crop_size
#     patch_w = crop_size
#     stride_depth = 8
#     stride_height = stride_size
#     stride_width = stride_size

#     N_patches_d = (img_d - patch_d) // stride_depth + 1
#     N_patches_h = (img_h - patch_h) // stride_height + 1
#     N_patches_w = (img_w - patch_w) // stride_width + 1
#     N_patches_img = N_patches_d * N_patches_h * N_patches_w

#     N_full_imgs = preds.shape[0] #多少块
    
#     full_prob = np.zeros((img_d, img_h, img_w, preds.shape[4]))  # Initialize to zero mega array with sum of Probabilities
#     full_sum = np.zeros((img_d, img_h, img_w, preds.shape[4]))
#     # print(full_prob.shape)
#     k = 0  # Iterator over all the patches
#     # print(preds[0].shape)
#     for d in range(N_patches_d):
#         for h in range(N_patches_h):
#             for w in range(N_patches_w):
#                 full_prob[d * stride_depth:(d * stride_depth) + patch_d,
#                 h * stride_height:(h * stride_height) + patch_h,
#                 w * stride_width:(w * stride_width) + patch_w, :] += preds[k]
#                 full_sum[d * stride_depth:(d * stride_depth) + patch_d,
#                 h * stride_height:(h * stride_height) + patch_h,
#                 w * stride_width:(w * stride_width) + patch_w, :] += 1
#                 k += 1
#     assert (k == preds.shape[0])
#     assert (np.min(full_sum) >= 1.0)  # At least one

#     final_avg = full_prob / full_sum
    
#     return final_avg







# def recompone_overlap(preds, crop_size, stride_size, img_h, img_w):
#     assert (len(preds.shape) == 4)  # 4D arrays

#     patch_h = crop_size
#     patch_w = crop_size
#     stride_height = stride_size
#     stride_width = stride_size

#     N_patches_h = (img_h - patch_h) // stride_height + 1
#     N_patches_w = (img_w - patch_w) // stride_width + 1
#     N_patches_img = N_patches_h * N_patches_w
#     #     print("N_patches_h: " +str(N_patches_h))
#     #     print("N_patches_w: " +str(N_patches_w))
#     #     print("N_patches_img: " +str(N_patches_img))
#     # assert (preds.shape[0]%N_patches_img==0)
#     N_full_imgs = preds.shape[0] // N_patches_img
#     #     print("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
#     full_prob = np.zeros(
#         (N_full_imgs, img_h, img_w, preds.shape[3]))  # itialize to zero mega array with sum of Probabilities
#     full_sum = np.zeros((N_full_imgs, img_h, img_w, preds.shape[3]))

#     k = 0  # iterator over all the patches
#     for i in range(N_full_imgs):
#         for h in range((img_h - patch_h) // stride_height + 1):
#             for w in range((img_w - patch_w) // stride_width + 1):
#                 full_prob[i, h * stride_height:(h * stride_height) + patch_h,
#                 w * stride_width:(w * stride_width) + patch_w, :] += preds[k]
#                 full_sum[i, h * stride_height:(h * stride_height) + patch_h,
#                 w * stride_width:(w * stride_width) + patch_w, :] += 1
#                 k += 1
#     #     print(k,preds.shape[0])
#     assert (k == preds.shape[0])
#     assert (np.min(full_sum) >= 1.0)  # at least one
#     final_avg = full_prob / full_sum
#     #     print('using avg')
#     return final_avg



def get_test_patches(img, crop_size, stride_size, rl=False):
    test_img = []

    test_img.append(img)
    test_img = np.asarray(test_img)

    #     test_img_adjust=img_process(test_img,rl=rl)
    test_img_adjust = test_img
    test_imgs = paint_border(test_img_adjust, crop_size, stride_size)

    test_img_patch = extract_patches(test_imgs, crop_size, stride_size)

    return test_img_patch, test_imgs.shape[1], test_imgs.shape[2], test_img_adjust

# 从给定的图像中提取出一系列的图像块（或称为 “patches”）
def extract_patches(full_imgs, crop_size, stride_size):
    patch_height = crop_size
    patch_width = crop_size
    stride_height = stride_size
    stride_width = stride_size

    assert (len(full_imgs.shape) == 4)  # 4D arrays
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image

    assert ((img_h - patch_height) % stride_height == 0 and (img_w - patch_width) % stride_width == 0)
    N_patches_img = ((img_h - patch_height) // stride_height + 1) * (
            (img_w - patch_width) // stride_width + 1)  # // --> division between integers
    N_patches_tot = N_patches_img * full_imgs.shape[0]

    patches = np.empty((N_patches_tot, patch_height, patch_width, full_imgs.shape[3]))
    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range((img_h - patch_height) // stride_height + 1):
            for w in range((img_w - patch_width) // stride_width + 1):
                patch = full_imgs[i, h * stride_height:(h * stride_height) + patch_height,
                        w * stride_width:(w * stride_width) + patch_width, :]
                patches[iter_tot] = patch
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches

# 处理图像的边界，以确保图像的高度和宽度可以被 crop_size 整除
def paint_border(imgs, crop_size, stride_size):
    patch_height = crop_size
    patch_width = crop_size
    stride_height = stride_size
    stride_width = stride_size

    assert (len(imgs.shape) == 4)
    img_h = imgs.shape[1]  # height of the full image
    img_w = imgs.shape[2]  # width of the full image
    leftover_h = (img_h - patch_height) % stride_height  # leftover on the h dim
    leftover_w = (img_w - patch_width) % stride_width  # leftover on the w dim
    full_imgs = None
    if (leftover_h != 0):  # change dimension of img_h
        tmp_imgs = np.zeros((imgs.shape[0], img_h + (stride_height - leftover_h), img_w, imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0], 0:img_h, 0:img_w, 0:imgs.shape[3]] = imgs
        full_imgs = tmp_imgs
    if (leftover_w != 0):  # change dimension of img_w
        tmp_imgs = np.zeros(
            (full_imgs.shape[0], full_imgs.shape[1], img_w + (stride_width - leftover_w), full_imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0], 0:imgs.shape[1], 0:img_w, 0:full_imgs.shape[3]] = imgs
        full_imgs = tmp_imgs
        #     print("new full images shape: \n" +str(full_imgs.shape))
        return full_imgs
    else:
        return imgs

# 将预测结果 pred 转换为图像块（或称为 “patches”）
def pred_to_patches(pred, crop_size, stride_size):
    return pred
    patch_height = crop_size
    patch_width = crop_size

    seg_num = 0
    #     print(pred.shape)

    assert (len(pred.shape) == 3)  # 3D array: (Npatches,height*width,2)

    pred_images = np.empty((pred.shape[0], pred.shape[1], seg_num + 1))  # (Npatches,height*width)
    pred_images[:, :, 0:seg_num + 1] = pred[:, :, 0:seg_num + 1]
    pred_images = np.reshape(pred_images, (pred_images.shape[0], patch_height, patch_width, seg_num + 1))
    return pred_images

# 处理图像块之间的重叠部分。由于每个像素位置可能被多个图像块覆盖，所以我们需要一种方法来合并这些图像块的预测结果。这个函数选择了取平均值的方法。
def recompone_overlap(preds, crop_size, stride_size, img_h, img_w):
    assert (len(preds.shape) == 4)  # 4D arrays

    patch_h = crop_size
    patch_w = crop_size
    stride_height = stride_size
    stride_width = stride_size

    N_patches_h = (img_h - patch_h) // stride_height + 1
    N_patches_w = (img_w - patch_w) // stride_width + 1
    N_patches_img = N_patches_h * N_patches_w
    #     print("N_patches_h: " +str(N_patches_h))
    #     print("N_patches_w: " +str(N_patches_w))
    #     print("N_patches_img: " +str(N_patches_img))
    # assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0] // N_patches_img
    #     print("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_prob = np.zeros(
        (N_full_imgs, img_h, img_w, preds.shape[3]))  # itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs, img_h, img_w, preds.shape[3]))

    k = 0  # iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_height + 1):
            for w in range((img_w - patch_w) // stride_width + 1):
                full_prob[i, h * stride_height:(h * stride_height) + patch_h,
                w * stride_width:(w * stride_width) + patch_w, :] += preds[k]
                full_sum[i, h * stride_height:(h * stride_height) + patch_h,
                w * stride_width:(w * stride_width) + patch_w, :] += 1
                k += 1
    #     print(k,preds.shape[0])
    assert (k == preds.shape[0])
    assert (np.min(full_sum) >= 1.0)  # at least one
    final_avg = full_prob / full_sum
    #     print('using avg')
    return final_avg