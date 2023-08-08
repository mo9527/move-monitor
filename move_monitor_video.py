#encoding=utf-8
import cv2
import time
from apscheduler.schedulers.background  import BackgroundScheduler
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from threading import Thread
from threading import Timer

# 全局变量
camera_fps=30
size=()
move_frame_cache = []
stop_flag = False


# 创建主窗口
window = ttk.Window()

# 第2步，给窗口的可视化起名字
window.title('监视器')
window.resizable(False, False)

def center_window(w, h):
    # 获取屏幕 宽、高
    ws = window.winfo_screenwidth()
    hs = window.winfo_screenheight()
    # 计算 x, y 位置
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)
    window.geometry('%dx%d+%d+%d' % (w, h, x, y))

center_window(380, 100)

def flush_cache_into_file():
    global move_frame_cache
    if len(move_frame_cache) < 10:
        return
    temp_frame_cache = move_frame_cache.copy()
    move_frame_cache = []
    video_file_name = str(time.strftime('%Y_%m_%d--%H_%M_%S', time.localtime(time.time())))
    # 定义编解码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'./video/{video_file_name}.mp4', fourcc, camera_fps, size)
    for frame in temp_frame_cache:
        out.write(frame)
    print('写入视频文件结束')
    out.release()


def begin_monitor():
    global size, camera_fps, stop_flag
    
    update_button_state(False)
    # 总是取前一帧做为背景（不用考虑环境影响）
    pre_frame = None
    # 帧率
    fps = 30
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
    camera_fps = camera.get(cv2.CAP_PROP_FPS)
    
    while True:
        if stop_flag:
            print('检测到关闭指令，退出循环')
            break
        
        start = time.time()
        # 读取视频流
        ret, frame = camera.read()
        # 转灰度图
        gray_lwpCV = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not ret:
            break
        end = time.time()

        #显示视频窗口
        cv2.imshow("capture", frame)
        # 帧图片中加入文字
        now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        cv2.putText(frame, str(now), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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

            for c in contours:
                # 设置敏感度
                # contourArea计算轮廓面积
                if cv2.contourArea(c) < 2000:
                    continue
                else:
                    print(f"{now} 运动帧出现")
                    # 画出矩形框架,返回值x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
                    (x, y, w, h) = cv2.boundingRect(c)
                    # rectangle(原图,(x,y)是矩阵的左上点坐标,(x+w,y+h)是矩阵的右下点坐标,(0,255,0)是画线对应的rgb颜色,2是所画的线的宽度)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cv2.imshow("capture", frame)
                    move_frame_cache.append(frame)
                    break
            pre_frame = gray_lwpCV
        if len(move_frame_cache) > 300:
            flush_cache_into_file()
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    print('camera 释放')
    cv2.destroyAllWindows()
    print('关闭视频窗口')
    stop_flag = False
    flush_cache_into_file()


monitor_t = Thread(target=begin_monitor, daemon=False, name='监视器线程')
def start():
    global monitor_t
    if monitor_t.ident is None:
        monitor_t.start()
    else:
        monitor_t = Thread(target=begin_monitor, daemon=False, name='监视器线程')
        monitor_t.start()
    
    
def destroy():
    global stop_flag
    stop_flag = True
    if monitor_t.ident is None:
        return
    update_button_state(True)
    

def update_button_state(enable = True):
    start_button.configure(state="normal" if enable else 'disabled')
    

start_button = ttk.Button(window, command=start, text='开始')
start_button.pack(padx=40, pady= 4, side=LEFT)

flush_button = ttk.Button(window, command=flush_cache_into_file, text='刷新', bootstyle=LIGHT)
flush_button.pack(padx=30, pady= 4, side=LEFT)

close_button = ttk.Button(window, command=destroy, text='关闭', bootstyle='danger')
close_button.pack(padx=30, pady= 4, side=LEFT)

def window_destroy():
    destroy()
    window.destroy()

if __name__ == '__main__':
    window.protocol('WM_DELETE_WINDOW', window_destroy)
    # 主窗口循环显示
    window.mainloop()
    