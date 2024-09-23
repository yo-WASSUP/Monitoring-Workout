# 导入YOLO模型和ai_gym模块
from ultralytics import YOLO
from ultralytics.solutions import ai_gym
# 导入OpenCV库
import cv2
# 导入os模块,用于文件操作
import os

# 加载YOLO模型,并指定模型权重文件为'yolov8m-pose.pt'
model = YOLO("yolov8m-pose.pt")

# 指定要处理的视频文件路径
video_path = "videos/pushup1.mp4"

# 使用OpenCV打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频文件是否成功打开
assert cap.isOpened(), "Error reading video file"

# 获取视频文件的宽度、高度和帧率
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 获取输入视频的文件名和扩展名
filename, ext = os.path.splitext(os.path.basename(video_path))

# 设置结果视频文件所在的文件夹名称
result_dir = "result_video"

# 创建result_video文件夹(如果不存在)
os.makedirs(result_dir, exist_ok=True)

# 构建输出视频的文件路径和名称
output_filename = f"inf_{filename}{ext}"
output_path = os.path.join(result_dir, output_filename)

# 创建视频写入器对象,用于保存处理后的视频
video_writer = cv2.VideoWriter(output_path,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))
# 初始化ai_gym模块
gym_object = ai_gym.AIGym()
# 设置ai_gym模块的参数
gym_object.set_args(line_thickness=2,
                    view_img=True,
                    pose_type="pushup", # pose_type="pullup",
                    kpts_to_check=[6, 8, 10])
# 初始化帧计数器
frame_count = 0
# 循环读取视频帧
while cap.isOpened():
    try:
        # 读取一帧视频数据
        success, im0 = cap.read()
        # 如果读取失败,则打印提示信息并退出循环
        if not success:
            print("video processing has been successfully completed.")
            break
        # 帧计数器加1
        frame_count += 1
        # 使用YOLO模型进行跟踪(推荐使用跟踪模式)
        results = model.track(im0, verbose=False)
        # 使用ai_gym模块进行运动计数
        im0 = gym_object.start_counting(im0, results, frame_count)
        # 将处理后的帧写入输出视频文件
        video_writer.write(im0)
    # 捕获异常,打印错误信息并继续执行下一次循环
    except Exception as e:
        print(f"An error occurred: {e}")
        continue

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
# 释放视频写入器资源
video_writer.release()