import torch  
import torch.nn as nn  
import torchvision.transforms as transforms  
import torchvision.models as models  
from PIL import Image  
  
# 确保PyTorch使用CUDA  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
# 加载你自己训练的ResNet50模型，并将其移到指定的设备上
model = models.resnet50()  # 初始化一个ResNet50模型

# 修改最后的全连接层  
num_original_classes = 1000  
num_new_classes = 2  
num_total_classes = num_original_classes + num_new_classes  

# 替换最后的全连接层  
model.fc = nn.Linear(model.fc.in_features, num_total_classes)

# 加载你自己训练的模型参数
model.load_state_dict(torch.load(r'D:\Desktop\AI\Image_AI\model.pth'))  
model = model.to(device)

# 设置模型为评估模式
model.eval()  
  
# 图像预处理流程  
transform = transforms.Compose([  
    transforms.Resize(256),  
    transforms.CenterCrop(224),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])  
  
# 加载并处理图像  
img_path = r'D:\Desktop\AI\Image_AI\Images\Lion_1.jpg'  
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
  
class_index_to_name = load_labels(r'D:\Desktop\AI\Image_AI\imagenet_labels\imagenet_labels_2.txt')

if len(class_index_to_name) != 1002:
    print(f"Warning: The number of classes in the label file is {len(class_index_to_name)}, but expected 1002.")
else:
    predicted_name = class_index_to_name[predicted.item()]  


print('Predicted class:', predicted_name)