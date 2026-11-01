from PIL import Image
from model2 import SimpleNeuralNetwork, SimpleCNN  
import torch
import torchvision.transforms as transforms
import os
import sys

# 1. Định nghĩa các nhãn (Classes) theo đúng thứ tự lúc train
CLASSES = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

# 2. Khởi tạo mô hình và nạp weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
weights_path = "save_model/model_lastcnn.pt"

if not os.path.exists(weights_path):
    print(f"❌ [LỖI]: Không tìm thấy file weights tại '{weights_path}'!")
    print("👉 Hãy chắc chắn bạn đã đặt file 'model_lastcnn.pt' vào thư mục 'save_model/' hoặc chạy training trước.")
    sys.exit(1)

checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint["model"])  
model.eval() 

# 3. Tiền xử lý ảnh đầu vào
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),          # Chuyển [0,255] → [0.0, 1.0], shape: (C, H, W)
    ]
)


def predict_image(image_path):
    """
    Dự đoán nhãn của ảnh từ đường dẫn cho trước.

    Args:
        image_path (str): Đường dẫn tới file ảnh cần dự đoán.

    Returns:
        str: Nhãn được dự đoán (ví dụ: "cat", "dog", ...).
    """
    if not os.path.exists(image_path):
        print(f"❌ [LỖI]: Không tìm thấy file ảnh tại '{image_path}'!")
        return None

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device) 

    with torch.no_grad():
        outputs = model(image_tensor)           
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item() * 100

    predicted_label = CLASSES[predicted_idx]
    print(f"🎯 Kết quả: Ảnh '{image_path}' được dự đoán là: [{predicted_label.upper()}] (Độ tin cậy: {confidence:.2f}%)")
    return predicted_label


if __name__ == "__main__":
    test_img = "test_cat.jpg"
    predict_image(test_img)
