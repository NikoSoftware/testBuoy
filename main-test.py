
from ultralytics import YOLO
import cv2

def run():
    model = YOLO("./runs/train/train/weights/best.pt")  # 加载自定义训练模型
    cap = cv2.VideoCapture(0)  # 0 表示默认摄像头，多摄像头可尝试1,2等[6,8](@ref)

    if not cap.isOpened():
        print("❌ 摄像头打开失败，请检查设备连接或权限")
        return

    print("▶️ 摄像头实时检测中（按 'q' 退出）...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 帧读取失败")
            break

        # 实时检测并显示结果
        results = model.predict(frame, conf=0.6, verbose=False)  # verbose=False关闭冗余输出
        annotated_frame = results[0].plot()  # 自动绘制检测框和标签[6](@ref)

        cv2.imshow("摄像头检测 - YOLOv5", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()