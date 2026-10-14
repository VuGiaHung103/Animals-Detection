from PIL import Image
from model2 import SimpleNeuralNetwork  # Import mô hình của bạn
import torch
import torchvision.transforms as transforms

# 1. Định nghĩa các nhãn (Classes) theo đúng thứ tự lúc train
CLASSES = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

# 2. Khởi tạo mô hình và nạp weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNeuralNetwork().to(device)

# Load file weights đã train
weights_path = "./checkpoints/animal_model_best.pth"
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()  # Chuyển sang chế độ dự đoán (evaluation mode)

# 3. Tiền xử lý ảnh đầu vào (sửa lại kích thước/transform cho giống lúc train)
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # Đổi lại kích thước phù hợp với model của bạn
        transforms.ToTensor(),
        # transforms.Normalize(...) # Nếu lúc train có dùng Normalize thì thêm vào đây
    ]
)


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Thêm batch dimension (1, C, H, W)

    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_idx = torch.argmax(outputs, dim=1).item()

    predicted_label = CLASSES[predicted_idx]
    print(f"Ảnh '{image_path}' được dự đoán là: {predicted_label}")
    return predicted_label


if __name__ == "__main__":
    # Test thử 1 bức ảnh bất kỳ
    predict_image("test_cat.jpg")