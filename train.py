from ultralytics import YOLO
import argparse

def train_model(data_yaml, model_weights='yolov8n-pose.pt', epochs=100, imgsz=640, batch=16, device='0'):
    """
    训练YOLOv8 Pose模型
    
    参数:
        data_yaml (str): 数据配置文件的路径
        model_weights (str): 预训练模型权重路径或名称
        epochs (int): 训练轮数
        imgsz (int): 输入图像尺寸
        batch (int): 批量大小
        device (str): 使用的设备，如'0'表示GPU 0，'cpu'表示CPU
    """
    # 加载模型
    model = YOLO(model_weights)  # 加载预训练模型
    
    # 训练模型
    results = model.train(
        data=data_yaml,
        iou=0.7,  # 直接设置IoU阈值 [0.5-0.95]
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=50,  # 早停耐心值
        project='robot_arm_pose',  # 项目名称
        name='exp',  # 实验名称
        exist_ok=True,  # 允许覆盖现有实验
        single_cls=True,  # 单类别模式
        optimizer='auto',  # 自动选择优化器
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率
        momentum=0.937,  # 动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3.0,  # 热身轮数
        warmup_momentum=0.8,  # 热身动量
        warmup_bias_lr=0.1,  # 热身偏置学习率
        box=7.5,  # 框损失权重
        cls=0.5,  # 分类损失权重
        dfl=1.5,  # dfl损失权重
        pose=12.0,  # 姿态损失权重
        kobj=2.0,  # 关键点对象损失权重
        nbs=64,  # 名义批量大小
        overlap_mask=True,  # 训练时掩码重叠
        mask_ratio=4,  # 掩码下采样比率
        dropout=0.0,  # 使用dropout正则化
        val=True,  # 训练期间验证
        save=True,  # 保存训练检查点和预测结果
        save_period=-1,  # 每x轮保存检查点
        deterministic=False,  # 确定性模式
        workers=8,  # 数据加载工作线程数
        verbose=True,  # 详细输出
        rect=False  # 必须为False才能使用自定义IoU
    )
    
    # 保存最终模型
    model.save('robot_arm_pose_final.pt')
    
    return results

if __name__ == "__main__":
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='Train YOLOv8 Pose model')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to data.yaml file')
    parser.add_argument('--weights', type=str, default='yolov8n-pose.pt', help='Pretrained model weights')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (e.g. 0 for GPU 0, cpu for CPU)')
    
    args = parser.parse_args()
    
    # 开始训练
    train_model(
        data_yaml=args.data,
        model_weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )