import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
import gradio as gr
from PIL import Image
import torch.nn.functional as F

# 加载数据集类别
full_ds = ImageFolder("bone_data")
class_names = full_ds.classes

# 加载模型
model = timm.create_model("convnextv2_nano.fcmae_ft_in22k_in1k", pretrained=True, num_classes=len(class_names))
model.eval()

def predict(image):
    if image is None:
        return "请上传一张骨折 X 光图片", {}
    
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    x = tf(image).unsqueeze(0)
    
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].numpy()
    
    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx] * 100)
    
    top3 = {class_names[i]: round(float(probs[i]*100), 2) for i in probs.argsort()[-3:][::-1]}
    
    emoji = "🟢" if confidence > 80 else "🟡" if confidence > 60 else "🔴"
    result_text = f"{emoji} **预测结果：{class_names[pred_idx]}**\n置信度：**{confidence:.1f}%**"
    
    return result_text, top3

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传骨折 X 光图片"),
    outputs=[
        gr.Markdown(label="AI 预测结果"),
        gr.Label(label="Top 3 预测概率")
    ],
    title="🦴 骨折类型智能分类系统",
    description="上传 X 光图片，AI 自动判断骨折类型",
    theme=gr.themes.Soft()
).launch()
