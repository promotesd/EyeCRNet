# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math

#使用三种椭圆结构元素/核去找瞳孔，
kernel_pup  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15, 15))
kernel_pup2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7))
kernel_cr   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))


def detect_pupil_from_thresholded(thresholded, size_limits=None, symmetry_tresh=0.5, fill_thresh=0.2, kernel=kernel_pup, window_name=None):
    '''
    thresholded:瞳孔已经标识出的阈值灰度图像
    size_limits:指定瞳孔候选区域的面积范围
    symmetry_tresh:对称性阈值，用于评估候选瞳孔的形状
    fill_thresh:填充阈值，用于比较候选区域的实际面积与其几何计算面积的接近程度
    kernel:用于形态学操作的核，影响腐蚀和膨胀的效果
    '''
    # 计算图像中心
    im_height, im_width = thresholded.shape
    center_x, center_y = im_width/2, im_height/2

    #使用开操作（先腐蚀再膨胀），再使用闭操作（先膨胀再腐蚀）
    blobs = cv2.morphologyEx(thresholded,cv2.MORPH_OPEN,kernel)
    blobs = cv2.morphologyEx(blobs,cv2.MORPH_CLOSE,kernel)

    # 如果window名字知道，可视化斑点
    if window_name:
        cv2.imshow(window_name, blobs)

    # 找检测斑点的轮廓；cv2.RETR_LIST检测所有轮廓，并返回轮廓列表；cv2.CHAIN_APPROX_NONE保存所有轮廓点
    blob_contours, hierarchy  = cv2.findContours(blobs,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    # Find pupil but checking one blob at the time. Pupils are round, so
    # add checks for 'roundness' criteria
    # If serveral blobs are found, select the one
    # closest to the center
    '''
    For a blob to be a pupil candidate
    1. blob must have the right area
    2. must be circular
    '''

    pupil_detected = False #初始化未找到轮廓
    old_distance_image_center = np.inf #旧的轮廓中心与图像中心距离
    for i, cnt in enumerate(blob_contours): #检测所有轮廓

        # 凸包处理轮廓，填补轮廓凹陷
        cnt = cv2.convexHull(cnt)

        # 仅仅轮廓点数量符合要求的轮廓被保留
        if len(cnt) < 10:
            continue

        # 计算轮廓面积与轮廓的外包矩形
        temp_area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        x, y, width, height = rect
        radius = 0.25 * (width + height)#计算平均半径
        r1,r2 = width,height
        if r1>r2: #调换位置，外包矩形的长和宽
            r1,r2 = r2,r1

        # 检查符合条件面积条件，对称性条件，填充度条件（检测出的面积与真实面积的）
        area_condition = True if size_limits is None else (size_limits[0] <= temp_area <= size_limits[1])
        symmetry_condition = (abs(1 - float(r1)/float(r2)) <= symmetry_tresh)
        fill_condition = (abs(1 - (temp_area / (np.pi * radius**2))) <= fill_thresh)

        # 如果条件满足，那么应该就是瞳孔
        if area_condition and symmetry_condition and fill_condition:
            # 计算斑点的矩
            moments = cv2.moments(cnt)

            # 通过矩计算斑点的质心，m10/m00表示x轴上的质心位置,m01/m00表示y轴上的位置
            cx, cy = moments['m10']/moments['m00'], \
                     moments['m01']/moments['m00']

            # 计算斑点中心与图像中心距离
            distance_image_center = np.sqrt((cx - center_x)**2 +
                                            (cy - center_y)**2)
            # Check if the current blob-center is closer
            # 如果刚刚检测的比前一个检测的距离更近，认定为新的瞳孔
            if distance_image_center < old_distance_image_center:
                pupil_detected = True

                # Store pupil variables
                contour_points = cnt
                area = temp_area

                cx_best = cx
                cy_best = cy

                # 将轮廓拟合椭圆
                ellipse = cv2.fitEllipse(contour_points.squeeze().astype('int'))
                (x_ellipse, y_ellipse), (MA, ma), angle = ellipse#MA主轴长，ma次轴长
                area_ellipse = np.pi / 4.0 * MA * ma

                old_distance_image_center = distance_image_center

    # If no potential pupil is found, due to e.g., blinks,
    # return nans
    if not pupil_detected:#如果为检测到，赋值nan
        cx_best = np.nan
        cy_best = np.nan
        area = np.nan
        contour_points = np.nan
        x_ellipse = np.nan
        y_ellipse = np.nan
        MA = np.nan
        ma = np.nan
        angle = np.nan
        area_ellipse = np.nan
#返回瞳孔特征
    pupil_features = {'cog':(cx_best, cy_best), 'area':area, 'contour_points': contour_points,
                      'ellipse' : ((x_ellipse, y_ellipse), (MA, ma), angle,
                                    area_ellipse)}
    return pupil_features
def detect_pupil(img, intensity_threshold, size_limits, window_name=None):
    ''' Identifies pupil blob
    Args:
        img - grayscale eye image 输入的眼睛灰度图像
        intensity_threshold - threshold used to find pupil area  用于二值化的阈值
        size_limite - [min_pupil_size, max_pupil_size] #检测出瞳孔大小的限制
        window_name - plots detected blobs is window #提供window name，检测到的瞳孔将会显示
                        name is given

    Returns:
        (cx, cy) - center of gravity binary pupil blob 二值化瞳孔斑点的中心
        area -  area of pupil blob 瞳孔斑点的面积
        countour_points - contour points of pupil 瞳孔的轮廓
        ellipse - parameters of ellipse fit to pupil blob #椭圆拟合参数
            (x_centre,y_centre),(minor_axis,major_axis),angle, area

    '''
    # cv2.THRESH_BINARY二值化类型，高于阈值为白色,ret是 函数返回的阈值，thresh 是二值化后的图像
    ret,thresh = cv2.threshold(img, intensity_threshold, 255, cv2.THRESH_BINARY)

    return detect_pupil_from_thresholded(thresh, size_limits=size_limits, window_name=window_name)

#%%
#从灰度眼睛图像中检测称为 CR（Corneal Reflections，角膜反射）的亮点
def detect_cr(img, intensity_threshold, size_limits,
              pupil_cr_distance_max, pup_center, no_cr=2, cr_img_size = (20,20),
              window_name=None):
    ''' Identifies cr blob (must be located below pupil center)
    Args:
        img - grayscale eye image 眼睛的灰色图像
        intensity_threshold - threshold used to find cr area(s) 二值化的图像阈值，用于分离角膜反射的区域
        size_limite - [min_cr_size, max_cr_size] 区域大小限制
        pupil_cr_distance_max - maximum allowed distance between pupil and CR 瞳孔与CR之间允许的最大距离
        no_cr - number of cr's to be found 被识别的CR的数量

    Returns:
        cr - cr featuers CR的特征

    '''
    #若找不到瞳孔中心的数据，直接返回CR的空列表
    if np.isnan(pup_center[0]):
        return [] 

    # 图像二值化
    ret,thresh1 = cv2.threshold(img, intensity_threshold,
                               255,cv2.THRESH_BINARY)

    # Close  holes in the cr, if any
    blobs = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel_cr)#3*3的椭圆核，先腐蚀后膨胀去除小噪点
    blobs = cv2.morphologyEx(blobs,cv2.MORPH_CLOSE,kernel_cr)#先膨胀后腐蚀，填充物体内的小洞

    if window_name:
        cv2.imshow(window_name, blobs)

    # 找轮廓，cv2.CHAIN_APPROX_SIMPLE缩水平方向、垂直方向和对角线方向的元素，只保留它们的终点坐标。
    blob_contours, hierarchy = cv2.findContours(blobs,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    cr = []

    #遍历每个轮廓
    for i, cnt in enumerate(blob_contours):
        # 轮廓点大于等于4不跳过循环
        if len(cnt) < 4:
            continue

        # 计算面积和外围矩形
        temp_area = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        x, y, width, height = rect
        radius = 0.25 * (width + height)
        r1,r2 = width,height
        if r1>r2:
            r1,r2 = r2,r1

        # 满足条件
        area_condition = (size_limits[0] <= temp_area <= size_limits[1])
        symmetry_condition = (abs(1 - float(r1)/float(r2)) <= 0.7)
        fill_condition = (abs(1 - (temp_area / (np.pi * radius**2))) <= 0.7)

        # If these criteria are fulfilled, a 角膜反射 is probably detected
        if area_condition and symmetry_condition and fill_condition:
            # Compute moments of blob
            moments = cv2.moments(cnt)

            # Compute blob center of gravity from the moments
            # Coordinate system (0, 0) upper left
            # \是续行符
            cx, cy = moments['m10']/moments['m00'], \
                     moments['m01']/moments['m00']

            # 计算当前CR中心与瞳孔中心的距离
            d = np.sqrt((cx - pup_center[0])**2 + (cy - pup_center[1])**2)
            #CR距离瞳孔过远
            if d > pupil_cr_distance_max:
                # print('cr too far away from pupil')
                continue
            #cy要在瞳孔上方
            if cy < (pup_center[1] - 0):
                # print('cr1 above pupil center')
                continue

            #若满足以上两个if，则填入x、y，area
            cr.append([cx, cy, temp_area])

    # 若选择的CR的数量超过预期的数量
    if len(cr) > no_cr:
        dist = []
        for c in cr: #对于每个CR，计算与瞳孔中心的距离
            dist.append(np.sqrt((c[0] - pup_center[0])**2 + \
                                (c[1] - pup_center[1])**2))

        # 排序且搜索最近的距离
        idx = np.argsort(dist)#根据大小进行排序，并得到索引
        cr = [cr[i] for i in idx[:no_cr]]#选择距离近的

    # If the correct number of cr's are detected,
    # distinguish between them using x-position, i.e.,
    # give them an identity, cr1, cr2, cr2, etc.
    #若CR的数量正确
    if len(cr) == no_cr:
        x_pos = []
        #加入这些对象
        for c in cr:
            x_pos.append(c[0])

        # 排序
        idx = np.argsort(x_pos)
        cr = [cr[i] for i in idx]

    return cr#返回排序后的


def img_cutout(img,pos,cutout_sz,mode=2,filler=0):
    '''
    img:输入的灰度图像
    pos:一个元组，表示截取区域的中心点坐标
    cutout_sz: 一个元组，表示截取区域的（width和height）
    mode:整数，用于指定如果截取区域超出图像边界时的处理方式。mode=2 表示用 filler 填充超出的部分。
    filler:用于填充超出边界部分的数值，默认为 0
    '''
    # Cut out image patch around pupil center location
    cx,cy = pos #截取图像中心的横纵坐标
    half_width  = cutout_sz[0]/2 #计算截取区域宽度的一半，用于确定截取框的横向边界
    half_height = cutout_sz[1]/2 #计算截取区域高度的一半，用于确定截取框的纵向边界
    im_height, im_width = img.shape  #图像大小

    # mode: either move cutout to fit on image (mode 1), or
    # replace parts of cutout beyond image with filler
    #初始化一个列表 padding 来记录需要在截取图像的每一侧添加的填充量。顺序分别是左、上、右、下。
    padding = [0,0,0,0] # left, top, right, bottom 

    #计算截取区域的横纵坐标范围
    x_range = [int(cx - half_width ), int(cx + half_width )] 
    y_range = [int(cy - half_height), int(cy + half_height)]

    # 确保截取区域不超出图像边界

    if x_range[0]<0: #检查截取区域是否超出图像的左边界
        x_range[0] = 0 #超出设为0
    sdiff = x_range[1]-x_range[0] #计算截取的实际宽度
    if x_range[0]==0 and sdiff<cutout_sz[0]: #左边界调整到图像最左边，未超出 且 实际截取宽度小于理想的截取区域宽度
        if mode==1:
            #这里通过增加右边界 x_range[1] 来扩展截取区域的宽度，确保截取区域的宽度等于预期宽度 cutout_sz[0]
            x_range[1] += (cutout_sz[0]-sdiff)
        else:
            #预期宽度和实际宽度之间的差值
            padding[0] = cutout_sz[0]-sdiff

    if y_range[0]<0:
        y_range[0] = 0
    sdiff = y_range[1]-y_range[0]
    if y_range[0]==0 and sdiff<cutout_sz[1]:
        if mode==1:
            y_range[1] += (cutout_sz[1]-sdiff)
        else:
            padding[1] = cutout_sz[1]-sdiff

    if x_range[1]>im_width:
        x_range[1] = im_width
    sdiff = x_range[1]-x_range[0]
    if x_range[1]==im_width and sdiff<cutout_sz[0]:
        if mode==1:
            x_range[0] -= (cutout_sz[0]-sdiff)
        else:
            padding[2] = cutout_sz[0]-sdiff

    if y_range[1]>im_height:
        y_range[1] = im_height
    sdiff = y_range[1]-y_range[0]
    if y_range[1]==im_height and sdiff<cutout_sz[1]:
        if mode==1:
            y_range[0] -= (cutout_sz[1]-sdiff)
        else:
            padding[3] = cutout_sz[1]-sdiff

    #cutout截取再x范围和y范围的图片
    cutout  = img[y_range[0] : y_range[1],
                  x_range[0] : x_range[1]]
    #off是记录矩形起始坐标
    off     = [x_range[0], y_range[0]]

    #对于padding列表中的非零元素，需要进行填充操作
    if any([p!=0 for p in padding]):

        #对于左位置不为零
        if padding[0]:
            #创建一个新的零矩阵，其高度与截取图像相同，宽度为所需的填充量。
            pad = np.zeros((cutout.shape[0],padding[0]),img.dtype)
            #若filler值不为0
            if filler!=0:
                #将pad中的元素均填为0
                pad[:,:] = filler
            #剪裁后的图片进行水平堆叠
            cutout = np.hstack((pad, cutout))
            #要填充左边，就要左边的起始坐标减去padding
            off[0] -= padding[0]
        #
        if padding[1]:
            pad = np.zeros((padding[1],cutout.shape[1]),img.dtype)
            if filler!=0:
                pad[:,:] = filler
            cutout = np.vstack((pad, cutout))
            off[1] -= padding[1]
        #右
        if padding[2]:
            pad = np.zeros((cutout.shape[0],padding[2]),img.dtype)
            if filler!=0:
                pad[:,:] = filler
            cutout = np.hstack((cutout, pad))
        #下
        if padding[3]:
            pad = np.zeros((padding[3],cutout.shape[1]),img.dtype)
            if filler!=0:
                pad[:,:] = filler
            cutout = np.vstack((cutout, pad))
        
    return cutout, off

def make_mask(ref_img,center,radii,angle=0,val=255,bg_val=0):
    '''
    ref_img : 参考的图像，确定掩码的尺寸与类型
    center : 圆或者椭圆的中心点坐标
    radii : 圆的半径，或者椭圆的半长轴和半短轴
    angle : 椭圆的旋转角度，默认为0
    val : 掩码中形状的像素值，默认为255（白色）
    bg_val : 背景像素值，默认为黑色
    '''


    subPixelFac = 8  #子像素因子，提高形状的绘制精度
    mask = np.zeros_like(ref_img)  #创建一个与参考图像尺寸和数据类型相同的全零数组。
    if bg_val!=0:    #如果背景值不为0，则将整个掩码数组设置为该背景值
        mask[:,:] = bg_val  


    #中心坐标乘以子像素因子 subPixelFac，并四舍五入到最近的整数，以提高精度。
    center = [int(np.round(x*subPixelFac)) for x in center]

    if not isinstance(radii,list):
        radii = [radii]
    radii  = [int(np.round(x*subPixelFac)) for x in radii]

    #如果 radii 长度为1，使用 cv2.circle 在掩码上绘制圆形。-1 表示填充圆形内部。
    #抗锯齿线型 (cv2.LINE_AA)
    if len(radii)==1:
        mask = cv2.circle (mask, center, radii[0]            , val, -1, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))
    else:

    #如果 radii 包含两个值，使用 cv2.ellipse 在掩码上绘制椭圆。参数包括中心点、半轴长度、旋转角度等。
        mask = cv2.ellipse(mask, center, radii, angle, 360, 0, val, -1, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))
    return mask