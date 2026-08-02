import os
import torch
import multiprocessing

# --- Import các file trong project ---
from src import config
from src.data_loader import get_data_loaders
from src.model import get_model
from src.train import train_model
from src.predict import predict_with_camera, predict_from_image


def get_image_path_from_terminal():
    """Yêu cầu người dùng nhập đường dẫn ảnh trực tiếp."""
    print("\n🖼️ Vui lòng dán đường dẫn đầy đủ tới file ảnh của bạn vào đây.")
    print("(Mẹo: Bạn có thể kéo và thả file ảnh vào cửa sổ này để lấy đường dẫn)")
    image_path = input("👉 Đường dẫn file ảnh: ")
    return image_path.strip().replace('"', '')


def main():
    multiprocessing.freeze_support()

    # Tải dữ liệu
    try:
        train_loader, val_loader, class_names, transform_predict = get_data_loaders(config)
        num_classes = len(class_names)
        print(f"🔍 Tìm thấy {num_classes} lớp: {class_names}")
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi khi tải dữ liệu: {e}")
        return

    # Lấy mô hình
    model = get_model(num_classes).to(config.DEVICE)

    # --- Kịch bản 1: ĐÃ CÓ MÔ HÌNH ---
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"✅ Tìm thấy mô hình đã huấn luyện tại: {config.MODEL_SAVE_PATH}")
        try:
            model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        except RuntimeError as e:
            print(f"❌ Lỗi khi tải mô hình: {e}")
            print("Mô hình cũ không tương thích với dữ liệu mới. Vui lòng xóa file và huấn luyện lại.")
            return

        print("📦 Đang tải mô hình phát hiện vật thể (YOLOv5)... Vui lòng chờ.")
        try:
            yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            yolo_model.to(config.DEVICE)
            print("✅ Tải YOLOv5 thành công!")
        except Exception as e:
            print(f"❌ Lỗi khi tải YOLOv5: {e}. Vui lòng kiểm tra kết nối internet.")
            return

        while True:
            print("\n" + "="*38)
            print(" LUA CHON CHUC NANG NHAN DIEN RAC THAI")
            print("-"*38)
            print("1. Nhận diện qua Camera trực tiếp 📸")
            print("2. Nhận diện từ một hình ảnh 🖼️")
            print("3. Thoát chương trình 🚪")2
            print("="*38)
            choice = input("👉 Vui lòng nhập lựa chọn của bạn (1, 2, hoặc 3): ")
            if choice == '1':
                predict_with_camera(model, yolo_model, transform_predict, config.DEVICE, class_names)
            elif choice == '2':
                image_path = get_image_path_from_terminal()
                if image_path and os.path.exists(image_path):
                    predict_from_image(model, yolo_model, transform_predict, config.DEVICE, class_names, image_path)
                else:
                    print("⚠️ Đường dẫn file không tồn tại hoặc bạn chưa nhập.")
            elif choice == '3':
                print("👋 Tạm biệt!")
                break
            else:
                print("❌ Lựa chọn không hợp lệ.")

    # --- Kịch bản 2: CHƯA CÓ MÔ HÌNH ---
    else:
        print("❌ Không tìm thấy mô hình đã huấn luyện. Bắt đầu huấn luyện.")
        train_model(model, train_loader, val_loader, config)
        print("\n🎉 Huấn luyện hoàn tất! Vui lòng chạy lại chương trình để sử dụng.")


if __name__ == '__main__':
    main()
