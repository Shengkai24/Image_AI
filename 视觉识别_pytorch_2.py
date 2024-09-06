import torch  
import torchvision.transforms as transforms  
import torchvision.models as models  
from PIL import Image  
  
# 确保PyTorch使用CUDA  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
# 加载预训练的ResNet50模型，并将其移到指定的设备上  
#model = models.resnet50(pretrained=True).to(device)
# 使用新的 weights 参数加载预训练的 ResNet50 模型  
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)  
model.eval() 
  
# 图像预处理流程  
transform = transforms.Compose([  
    transforms.Resize(256),  
    transforms.CenterCrop(224),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])  
  
# 加载并处理图像  
img_path = r'D:\Desktop\AI\Image_AI\Training_material\Car_1.png'  
try:  
    img = Image.open(img_path)
    img = img.convert('RGB')
except FileNotFoundError:  
    print(f"The image at {img_path} could not be found.")  
    exit()  
  
img_t = transform(img).unsqueeze(0).to(device)  
  
# 进行预测  
with torch.no_grad():  
    output = model(img_t)  
    _, predicted = torch.max(output, 1)  
  
# 打印预测分数和预测的类别索引  
print('Predicted scores:', output.cpu().numpy())   
  
# 加载标签文件并获取预测的类别名称  
def load_labels(labels_file):  
    try:  
        with open(labels_file) as reader:  
            labels = [line.strip() for line in reader]  
        return labels  
    except FileNotFoundError:  
        print(f"The labels file at {labels_file} could not be found.")  
        exit()  
  
class_index_to_name = load_labels(r'D:\Desktop\AI\Image_AI\imagenet_labels\imagenet_labels.txt')
predicted_name = class_index_to_name[predicted.item()]  
  
print('Predicted class:', predicted_name)