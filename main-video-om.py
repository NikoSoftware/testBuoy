import cv2
import numpy as np
import acl
import time
import ctypes


class YOLOv11_NPU:
    def __init__(self, model_path):
        # 增强错误处理
        ret = acl.init()
        if ret != 0:
            raise RuntimeError(f"ACL初始化失败: 错误码{ret}")

        self.device_id = 0
        ret = acl.rt.set_device(self.device_id)
        if ret != 0:
            acl.finalize()
            raise RuntimeError(f"设置设备失败: 错误码{ret}")

        # 加载OM模型
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        if ret != 0:
            acl.rt.reset_device(self.device_id)
            acl.finalize()
            raise RuntimeError(f"模型加载失败: 错误码{ret}")

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            acl.mdl.unload(self.model_id)
            acl.rt.reset_device(self.device_id)
            acl.finalize()
            raise RuntimeError(f"获取模型描述失败: 错误码{ret}")

        # 获取输入输出大小
        self.input_size = acl.mdl.get_input_size_by_index(self.model_desc, 0)
        self.output_size = acl.mdl.get_output_size_by_index(self.model_desc, 0)
        print(f"输入大小: {self.input_size}字节, 输出大小: {self.output_size}字节")

        # 内存分配（关键修复）
        self.input_buffer, ret = acl.rt.malloc(self.input_size, 0)  # 0=ACL_MEM_MALLOC_HUGE_FIRST
        if ret != 0 or self.input_buffer == 0:
            self._release_resources()
            raise RuntimeError(f"输入内存分配失败: 错误码{ret}")

        self.output_buffer, ret = acl.rt.malloc(self.output_size, 0)
        if ret != 0 or self.output_buffer == 0:
            self._release_resources()
            raise RuntimeError(f"输出内存分配失败: 错误码{ret}")

        # 创建流
        self.stream, ret = acl.rt.create_stream()
        if ret != 0:
            self._release_resources()
            raise RuntimeError(f"创建流失败: 错误码{ret}")

        print("NPU引擎初始化成功")

    def _release_resources(self):
        """统一释放资源"""
        if hasattr(self, 'input_buffer') and self.input_buffer:
            acl.rt.free(self.input_buffer)
        if hasattr(self, 'output_buffer') and self.output_buffer:
            acl.rt.free(self.output_buffer)
        if hasattr(self, 'model_id'):
            acl.mdl.unload(self.model_id)
        if hasattr(self, 'stream'):
            acl.rt.destroy_stream(self.stream)
        acl.rt.reset_device(self.device_id)
        acl.finalize()

    def preprocess(self, frame):
        """修复：内存对齐优化 & 错误处理"""
        if frame is None or frame.size == 0:
            raise ValueError("输入帧为空")

        # 强制内存连续
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        h, w = frame.shape[:2]
        scale = min(640 / w, 640 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(frame, (new_w, new_h))

        # 填充至640x640
        pad_x = (640 - new_w) // 2
        pad_y = (640 - new_h) // 2
        img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 转换格式并强制内存对齐
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]  # [1,3,640,640]

        # 关键修复：使用np.require确保内存对齐
        img = np.require(img, requirements=['C_CONTIGUOUS', 'ALIGNED'])

        # 检查内存大小
        if img.nbytes > self.input_size:
            raise MemoryError(f"输入数据大小({img.nbytes}B)超过模型预期({self.input_size}B)")

        # 数据拷贝
        ret = acl.rt.memcpy(self.input_buffer, 0, img.ctypes.data,
                            img.nbytes, 0)  # 0=HOST_TO_DEVICE
        if ret != 0:
            raise RuntimeError(f"数据拷贝失败(错误码:{ret})")

        # 等待拷贝完成
        acl.rt.synchronize_stream(self.stream)

        self.scale = scale
        self.padding = (pad_x, pad_y)
        return img.shape[2:]

    def infer_sync(self):
        """同步推理（阻塞式）"""
        # 创建数据集缓冲区
        input_dataset = acl.mdl.create_dataset_buffer(self.input_buffer)
        output_dataset = acl.mdl.create_dataset_buffer(self.output_buffer)

        # 执行推理
        ret = acl.mdl.execute(self.model_id, [input_dataset], [output_dataset])
        if ret != 0:
            raise RuntimeError(f"推理执行失败: 错误码{ret}")

        # 阻塞等待完成
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != 0:
            raise RuntimeError(f"同步流失败: 错误码{ret}")

        # 取回输出数据 [1,6,8400]
        host_output = np.zeros((1, 6, 8400), dtype=np.float32)
        ret = acl.rt.memcpy(host_output.ctypes.data, self.output_buffer,
                            self.output_size, 1)  # 1=DEVICE_TO_HOST
        if ret != 0:
            raise RuntimeError(f"输出拷贝失败: 错误码{ret}")
        return host_output

    # ...（decode_predictions、diou_nms、calculate_diou、transform_boxes方法保持不变）...

    def release(self):
        """释放NPU资源"""
        self._release_resources()


# ===== 视频处理主流程 =====
if __name__ == "__main__":
    # 使用用户提供的路径
    video_path = "./datasets/video/cat_blibli_7.mp4"
    model_path = "./runs/train/train/weights/best.om"

    try:
        print("正在初始化NPU引擎...")
        infer_engine = YOLOv11_NPU(model_path)

        print("正在打开视频文件...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        # 获取视频参数
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频信息: {frame_width}x{frame_height}, FPS: {fps:.1f}, 总帧数: {total_frames}")

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

        # 性能监控
        frame_count = 0
        start_time = time.time()
        last_log_time = time.time()

        print("开始处理视频...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            orig_shape = frame.shape

            try:
                # 1. 预处理
                infer_engine.preprocess(frame)

                # 2. NPU推理
                output = infer_engine.infer_sync()

                # 3. 后处理
                boxes = infer_engine.decode_predictions(output, conf_thres=0.5)
                if len(boxes) > 0:
                    boxes = infer_engine.diou_nms(boxes)
                    boxes = infer_engine.transform_boxes(boxes, orig_shape)

                # 4. 可视化
                for box in boxes:
                    x1, y1, x2, y2, conf, cls_id = box
                    label = f"{cls_id}:{conf:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 5. 显示帧率和进度
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time
                progress = frame_count / total_frames * 100

                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"进度: {progress:.1f}%", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # 写入输出视频
                out.write(frame)

                # 每5秒打印一次进度
                if time.time() - last_log_time > 5:
                    print(f"已处理 {frame_count}/{total_frames} 帧 ({progress:.1f}%), FPS: {current_fps:.1f}")
                    last_log_time = time.time()

                # 显示结果
                cv2.imshow('YOLOv11 NPU Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"处理第 {frame_count} 帧时出错: {str(e)}")
                # 跳过当前帧继续处理
                continue

    except Exception as e:
        print(f"程序运行出错: {str(e)}")
    finally:
        print("正在释放资源...")
        if 'infer_engine' in locals():
            infer_engine.release()
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals():
            out.release()
        cv2.destroyAllWindows()
        if frame_count > 0:
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"处理完成! 总帧数: {frame_count}, 总耗时: {total_time:.1f}秒, 平均FPS: {avg_fps:.1f}")