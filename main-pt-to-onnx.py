from ultralytics import YOLO
import onnx  # 添加ONNX库支持


def run():
    # 加载预训练模型
    model = YOLO("./runs/train/train/weights/best.pt")


    # ---- 添加ONNX导出部分 ----
    # 导出ONNX模型（关键参数说明见注释）
    export_path = model.export(
        format="onnx",  # 指定导出格式
        imgsz=(640, 640),  # 输入尺寸需与训练一致[6,8](@ref)
        dynamic=False,  # 禁用动态输入尺寸（部署兼容性更好）[6,8](@ref)
        simplify=True,  # 启用模型简化（移除冗余节点）[6](@ref)
        opset=12,  # 推荐使用12+版本（支持最新算子）[3,6](@ref)
        batch=1,  # 部署常用批处理大小[6](@ref)
        device="cpu",  # 在CPU上导出（避免CUDA依赖）[6](@ref)
        workspace=4.0  # 为TensorRT优化预留空间[6](@ref)
    )
    print(f"✅ ONNX模型已导出至: {export_path}")

    # 验证ONNX模型有效性
    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX模型验证通过")


if __name__ == '__main__':
    run()