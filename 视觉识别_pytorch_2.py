import os
import torch  
import torch.nn as nn  
import torchvision.models as models  
import torchvision.transforms as transforms  
from torchvision.datasets import ImageFolder  
from torch.utils.data import DataLoader  

# 1. 加载预训练的ResNet50模型  
model = models.resnet50(pretrained=True)  

# 打印原始的全连接层信息
print(model.fc)

# 修改最后的全连接层  
num_original_classes = 1000  
num_new_classes = 2  
num_total_classes = num_original_classes + num_new_classes  

# 冻结模型的所有层，除了最后一层
for param in model.parameters():
    param.requires_grad = False

# 替换最后的全连接层，输出改为 1002 类
model.fc = nn.Linear(model.fc.in_features, num_total_classes)

# 现在只训练最后的全连接层
for param in model.fc.parameters():
    param.requires_grad = True

# 再次打印新的全连接层信息以确认替换成功
print(model.fc)

# 加载标签文件并获取预测的类别名称  
def load_labels(labels_file):  
    try:  
        with open(labels_file) as reader:  
            labels = [line.strip() for line in reader]  
        return labels  
    except FileNotFoundError:  
        print(f"The labels file at {labels_file} could not be found.")  
        exit()  

labels = load_labels(os.getenv('LABELS_FILE', r'D:\Desktop\AI\Image_AI\imagenet_labels\imagenet_labels_2.txt'))

# 2. 准备你的数据  
data_root = os.getenv('DATA_ROOT', r'D:\Desktop\AI\Image_AI\Images')  # 使用环境变量或配置文件来设置数据路径
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# 使用新类的数据进行训练
train_dataset = ImageFolder(root=os.path.join(data_root, 'train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 保持验证集中的两类
val_dataset = ImageFolder(root=os.path.join(data_root, 'val'), transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 3. 定义损失函数和优化器  
def initialize_model(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.0001, momentum=0.9)
    return criterion, optimizer

criterion, optimizer = initialize_model(model)

# 4. 训练模型  
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
except Exception as e:
    print(f"Error setting up device: {e}")
    # 尝试使用CPU继续运行
    device = torch.device("cpu")
    model = model.to(device)
    print("Falling back to CPU.")

num_epochs = 10  

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    # 验证阶段  
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# 打印预测分数和预测的类别索引  
print('Predicted scores:', outputs.cpu().numpy())

for idx in predicted:
    print(f'Predicted class index: {idx}, class name: {labels[idx]}')


# 保存模型  
model_save_path = os.getenv('MODEL_SAVE_PATH', r'D:\Desktop\AI\Image_AI\model.pth')
torch.save(model.state_dict(), model_save_path)
