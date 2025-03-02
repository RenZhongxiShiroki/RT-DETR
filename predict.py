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

# 加载模型 根据你选择的模型文件
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
backbone = PResNet(depth=18, variant='d', freeze_at=-1, return_idx=[1, 2, 3], num_stages=4, freeze_norm=True,
                   pretrained=False).to(device)
encoder = HybridEncoder(in_channels=[128, 256, 512], feat_strides=[8, 16, 32], hidden_dim=256, use_encoder_idx=[2],
                        num_encoder_layers=1,
                        nhead=8, dim_feedforward=1024, dropout=0., enc_act='gelu', pe_temperature=10000, expansion=0.5,
                        depth_mult=1, act='silu', eval_spatial_size=[640, 640]).to(device)
decoder = RTDETRTransformer(num_classes=6, feat_channels=[256, 256, 256], feat_strides=[8, 16, 32], hidden_dim=256,
                            num_levels=3, num_queries=300, num_decoder_layers=3, num_denoising=100, eval_idx=-1,
                            eval_spatial_size=[640, 640]).to(device)
rtdetr = RTDETR(backbone=backbone, encoder=encoder, decoder=decoder,
                multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]).to(device)

postprocessor = RTDETRPostProcessor(num_classes=6)
orig = torch.full((1, 2), 640).to(device)

# 加载图片
transform1 = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()  # range [0, 255] -> [0.0,1.0]
]
)
x = Image.open("D:\zhangsan\work\deep_learning\image_data\X\\a(67).jpg").convert('RGB')
x = transform1(x)
x = x.unsqueeze(0).cuda()

# 权重加载
weights = torch.load("D:\zhangsan\my_code\origin_RT-DETR\RT-DETR-main\\rtdetr_pytorch\\tools\output\\rtdetr_r18vd_6x_coco\checkpoint.pth")
rtdetr.load_state_dict(weights['model'])
rtdetr.eval()

# 模型推理
y = rtdetr(x)

# 结果处理
result = postprocessor(y, orig)

lables = result[0]['labels']
boxes = result[0]['boxes']
scores = result[0]['scores']
# 图片展示

image = cv2.imread('D:\zhangsan\work\deep_learning\image_data\X\\a(67).jpg')
image = cv2.resize(image, (640, 640))
for label, box, score in zip(lables, boxes, scores):
    if score > 0.5:  # 给定阈值过滤目标框
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 绘制图像，将CV的BGR换成RGB
plt.show()  # 显示图像
