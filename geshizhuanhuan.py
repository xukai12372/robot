import json
import os
from tqdm import tqdm  # 用于显示进度条

def convert_coco_to_yolo(coco_json_path: str, output_dir: str, image_width: int, image_height: int) -> None:
    """
    将COCO格式的关键点标注转换为YOLOv8 Pose格式
    
    参数:
        coco_json_path: COCO JSON文件路径
        output_dir: YOLO格式标注输出目录
        image_width: 图像宽度（用于归一化）
        image_height: 图像高度（用于归一化）
    """
    try:
        # 加载COCO JSON文件
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 预处理：按image_id分组annotations（提升性能）
        annotations_dict = {}
        for ann in coco_data['annotations']:
            if 'keypoints' not in ann:
                continue  # 跳过无关键点的标注
            img_id = ann['image_id']
            if img_id not in annotations_dict:
                annotations_dict[img_id] = []
            annotations_dict[img_id].append(ann)
        
        # 处理每张图片
        for image_data in tqdm(coco_data['images'], desc="Converting COCO to YOLO"):
            image_id = image_data['id']
            image_name = image_data['file_name']
            
            # 获取当前图片的所有关键点标注
            if image_id not in annotations_dict:
                continue  # 跳过无标注的图片
            
            # 准备YOLO格式内容
            yolo_lines = []
            for ann in annotations_dict[image_id]:
                keypoints = ann['keypoints']
                
                # 计算边界框（从关键点生成）
                kps_x = keypoints[0::3]  # 所有x坐标
                kps_y = keypoints[1::3]  # 所有y坐标
                kps_v = keypoints[2::3]  # 所有可见性标志
                
                # 过滤不可见关键点（v=0）
                visible_x = [x for x, v in zip(kps_x, kps_v) if v > 0]
                visible_y = [y for y, v in zip(kps_y, kps_v) if v > 0]
                
                if not visible_x or not visible_y:
                    continue  # 跳过无可视关键点的标注
                
                # 计算边界框
                x_min, x_max = min(visible_x), max(visible_x)
                y_min, y_max = min(visible_y), max(visible_y)
                
                # 归一化坐标
                x_center = (x_min + x_max) / (2 * image_width)
                y_center = (y_min + y_max) / (2 * image_height)
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height
                
                # 构建YOLO行（类别ID + 边界框 + 关键点）
                yolo_line = [f"{0} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"]
                
                # 添加关键点（x,y,v）
                for x, y, v in zip(kps_x, kps_y, kps_v):
                    yolo_line.append(f"{x/image_width:.6f} {y/image_height:.6f} {v}")
                
                yolo_lines.append(" ".join(yolo_line))
            
            # 跳过无有效标注的图片
            if not yolo_lines:
                continue
            
            # 写入YOLO标注文件
            annotation_file_name = os.path.splitext(image_name)[0] + '.txt'
            annotation_file_path = os.path.join(output_dir, annotation_file_name)
            
            with open(annotation_file_path, 'w') as f:
                f.write("\n".join(yolo_lines))
        
        print(f"\n转换完成！结果保存在: {output_dir}")
    
    except Exception as e:
        print(f"转换失败: {str(e)}")
        raise

# 示例用法
if __name__ == "__main__":
    # 输入参数
    coco_json_path = "D:\\Robotic_Arm_Keypoint_Tracking_YoloV8-Pose-main\\arm1\\annotations\\person_keypoints_default.json"
    output_dir = "D:\\Robotic_Arm_Keypoint_Tracking_YoloV8-Pose-main\\val\\labels"
    image_size = 640  # YOLOv8推荐尺寸
    
    # 打印验证路径
    print(f"COCO JSON路径: {coco_json_path}")
    print(f"输出目录: {output_dir}")
    
    # 运行转换
    convert_coco_to_yolo(
        coco_json_path=coco_json_path,
        output_dir=output_dir,
        image_width=image_size,
        image_height=image_size
    )