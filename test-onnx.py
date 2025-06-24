from ultralytics import YOLO
import cv2

# 使用摄像头实时检测
if __name__ == '__main__':
    # 加载ONNX模型（确保OpenCV已安装ONNX支持）
    model = YOLO('./runs/train/train/weights/best.onnx')

    # 初始化摄像头（0=默认摄像头，1/2=外接摄像头）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 摄像头打开失败，请检查设备连接或权限")
        exit()

    print("▶️ 摄像头实时检测中（按 'q' 退出）...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 帧读取失败")
            break

        # 执行实时检测
        results = model(frame, verbose=False)  # 关闭冗余输出

        # 获取并可视化检测结果
        annotated_frame = results[0].plot()  # 自动绘制检测框和标签

        # 显示实时画面
        cv2.imshow('YOLO Real-time Detection', annotated_frame)

        # 打印检测结果（可选）
        print("检测结果:", results[0].boxes)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()