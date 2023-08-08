import cv2
import time

# 保存截图
save_path = './img/'

# 定义摄像头对象，其参数0表示第一个摄像头
camera = cv2.VideoCapture(0)

# 判断视频是否打开
print('摄像头开启' if camera.isOpened() else '摄像头未打开')

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)  # 设置宽度
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 设置高度

# 测试用,查看视频size
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('size:' + repr(size))

# 帧率
fps = 5
# 总是取前一帧做为背景（不用考虑环境影响）
pre_frame = None
file_name_formatter = '%Y_%m_%d_%H_%M_%S'


while True:
    start = time.time()
    # 读取视频流
    ret, frame = camera.read()
    # 转灰度图
    gray_lwpCV = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not ret:
        break
    end = time.time()

    cv2.imshow("capture", frame)

    # 运动检测部分
    seconds = end - start
    if seconds < 1.0 / fps:
        time.sleep(1.0 / fps - seconds)
    # gray_lwpCV = cv2.resize(gray_lwpCV, (size[0], size[1]))
    # 用高斯滤波进行模糊处理
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

    # 如果没有背景图像就将当前帧当作背景图片
    if pre_frame is None:
        pre_frame = gray_lwpCV
    else:
        # absdiff把两幅图的差的绝对值输出到另一幅图上面来
        img_delta = cv2.absdiff(pre_frame, gray_lwpCV)
        # threshold阈值函数(原图像应该是灰度图,对像素值进行分类的阈值,当像素值高于（有时是小于）阈值时应该被赋予的新的像素值,阈值方法)
        thresh = cv2.threshold(img_delta, 25, 255, cv2.THRESH_BINARY)[1]
        # 膨胀图像
        thresh = cv2.dilate(thresh, None, iterations=2)
        # findContours检测物体轮廓(寻找轮廓的图像,轮廓的检索模式,轮廓的近似办法)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pre_jpg_name = ''
        for c in contours:
            # 设置敏感度
            # contourArea计算轮廓面积
            if cv2.contourArea(c) < 1000:
                continue
            else:
                print("出现目标物，请求核实")
                # 画出矩形框架,返回值x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
                (x, y, w, h) = cv2.boundingRect(c)
                # rectangle(原图,(x,y)是矩阵的左上点坐标,(x+w,y+h)是矩阵的右下点坐标,(0,255,0)是画线对应的rgb颜色,2是所画的线的宽度)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # putText 图片中加入文字
                now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                jpg_name = str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))

                # 一秒只保存一张，不覆盖
                if jpg_name == pre_jpg_name:
                    continue

                pre_jpg_name = jpg_name

                cv2.imshow("capture", frame)
                cv2.putText(frame, str(now), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 保存图像
                img_file = save_path + jpg_name + '.jpg'
                print("保存图像", img_file)
                cv2.imwrite(img_file, frame)
                break
        pre_frame = gray_lwpCV

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

