# 导包
import matplotlib.pyplot as plt
from src.nn.backbone.presnet import PResNet
from src.zoo.rtdetr import RTDETR
from src.zoo.rtdetr import HybridEncoder
from src.zoo.rtdetr import RTDETRTransformer
from src.zoo.rtdetr import RTDETRPostProcessor
import torch
from PIL import Image
from torchvision import transforms
import cv2
import os

# 创建结果保存文件夹
output_dir = "D:\zhangsan\work\github_code\RT-DETR-origin\RT-DETR\\rtdetr_pytorch\img_outr"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载模型 根据你选择的模型文件
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
backbone = PResNet(depth=50, variant='d', freeze_at=-1, return_idx=[1, 2, 3], num_stages=4, freeze_norm=True,
                   pretrained=False).to(device)
encoder = HybridEncoder(in_channels=[512, 1024, 2048], feat_strides=[8, 16, 32], hidden_dim=256, use_encoder_idx=[2],
                        num_encoder_layers=1,
                        nhead=8, dim_feedforward=1024, dropout=0., enc_act='gelu', pe_temperature=10000, expansion=1.0,
                        depth_mult=1, act='silu', eval_spatial_size=[640, 640]).to(device)
decoder = RTDETRTransformer(num_classes=5, feat_channels=[256, 256, 256], feat_strides=[8, 16, 32], hidden_dim=256,
                            num_levels=3, num_queries=300, num_decoder_layers=6, num_denoising=100, eval_idx=-1,
                            eval_spatial_size=[640, 640]).to(device)
rtdetr = RTDETR(backbone=backbone, encoder=encoder, decoder=decoder,
                multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]).to(device)

postprocessor = RTDETRPostProcessor(num_classes=6)
orig = torch.full((1, 2), 640).to(device)

# 权重加载
weights = torch.load(
    "D:\zhangsan\work\github_code\RT-DETR-origin\RT-DETR\\rtdetr_pytorch\\tools\output\\rtdetr_r50vd_6x_coco_0\checkpoint0069.pth")
rtdetr.load_state_dict(weights['model'])
rtdetr.eval()

# 定义图像处理
transform1 = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()  # range [0, 255] -> [0.0, 1.0]
])

# 处理文件夹中的所有图像
input_dir = "D:\zhangsan\work\github_code\RT-DETR-origin\RT-DETR\\rtdetr_pytorch\img_voc/"
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 只处理图片文件
        img_path = os.path.join(input_dir, filename)

        # 加载图片
        x = Image.open(img_path).convert('RGB')
        x = transform1(x)
        x = x.unsqueeze(0).to(device)

        # 模型推理
        with torch.no_grad():
            y = rtdetr(x)

        # 结果处理
        result = postprocessor(y, orig)
        lables = result[0]['labels']
        boxes = result[0]['boxes']
        scores = result[0]['scores']

        # 读取并调整原始图片
        image = cv2.imread(img_path)
        image = cv2.resize(image, (640, 640))

        # 在图片上绘制预测框
        for label, box, score in zip(lables, boxes, scores):
            if score > 0.5:  # 过滤低置信度目标框
                x1, y1, x2, y2 = box
                # 绘制边界框
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                # 添加置信度文本
                confidence_text = f"{score:.2f}"
                cv2.putText(image, confidence_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            1)

        # 保存处理后的图片到 result 文件夹
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)
        print(f"Processed and saved: {output_path}")

# 完成后提示
print("All images processed and saved.")
