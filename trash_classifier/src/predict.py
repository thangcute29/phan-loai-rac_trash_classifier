import torch
import cv2
from PIL import Image


def test_model(model, test_loader, device):
    """
    Hàm test model trên tập test (nếu cần).
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total if total > 0 else 0
    print(f"✅ Độ chính xác trên tập test: {acc:.2f}%")
    return acc


def predict_with_camera(model, yolo_model, transform, device, class_names):
    """
    Sử dụng mô hình để dự đoán loại rác thải từ camera.
    """
    print("📸 Bắt đầu chế độ dự đoán bằng camera. Nhấn 'q' để thoát.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở camera. Vui lòng kiểm tra lại thiết bị.")
        return

    model.eval()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > 0.25:  # Ngưỡng tin cậy
                x1, y1, x2, y2 = map(int, xyxy)

                cropped_img = frame[y1:y2, x1:x2]

                if cropped_img.size > 0:
                    img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                    img_tensor = transform(img_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        _, predicted_idx = torch.max(outputs, 1)
                        predicted_class = class_names[predicted_idx.item()]

                    label = predicted_class
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Camera - Nhận diện rác thải', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn q để thoát
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Đã tắt camera.")


def predict_from_image(model, yolo_model, transform, device, class_names, image_path):
    """
    Sử dụng mô hình để dự đoán loại rác thải từ một tệp hình ảnh.
    """
    print(f"🖼️ Bắt đầu dự đoán từ hình ảnh: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Lỗi: Không thể đọc được hình ảnh từ đường dẫn: {image_path}")
        return

    model.eval()
    results = yolo_model(frame)
    detection_count = 0

    # --- Giai đoạn 1: Phát hiện vật thể ---
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > 0.25:
            detection_count += 1
            x1, y1, x2, y2 = map(int, xyxy)
            cropped_img = frame[y1:y2, x1:x2]

            if cropped_img.size > 0:
                img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                img_tensor = transform(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    _, predicted_idx = torch.max(outputs, 1)
                    predicted_class = class_names[predicted_idx.item()]

                label = predicted_class
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # --- Giai đoạn 2: Nếu YOLO không phát hiện ---
    if detection_count == 0:
        print("🤷 Không phát hiện được vật thể. Phân loại toàn bộ ảnh...")
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = class_names[predicted_idx.item()]

        label = predicted_class
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        print(f"✅ Phát hiện và phân loại {detection_count} đối tượng.")

    # Hiển thị ảnh kết quả
    cv2.imshow("Kết quả dự đoán", frame)
    print("ℹ️ Nhấn phím bất kỳ để đóng cửa sổ hình ảnh.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
