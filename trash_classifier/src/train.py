import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

def train_model(model, train_loader, val_loader, config):
    # Loss với label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Chỉ cập nhật những params requires_grad=True (nếu bạn freeze một số layer)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config.LEARNING_RATE, weight_decay=1e-4)

    # ReduceLROnPlateau: không dùng tham số 'verbose' để tránh lỗi trên một số phiên bản PyTorch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.0
    patience = 8
    no_improve = 0

    # Cố gắng lấy tên lớp để in classification_report (nếu có)
    class_names = None
    try:
        class_names = val_loader.dataset.dataset.classes
    except Exception:
        try:
            class_names = val_loader.dataset.classes
        except Exception:
            class_names = None

    print("🚀 Bắt đầu huấn luyện...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", unit="batch")

        for inputs, labels in pbar:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=(running_loss / total), acc=(100 * correct / total))

        epoch_acc = 100 * correct / total if total > 0 else 0.0

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
        print(f"\n✅ Epoch {epoch+1}: Train Acc: {epoch_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Lưu model tốt nhất
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"⭐ Đã lưu mô hình tốt nhất với độ chính xác: {best_acc:.2f}%")
            no_improve = 0
        else:
            no_improve += 1

        # scheduler step dựa trên val_acc
        try:
            scheduler.step(val_acc)
        except Exception:
            # fallback: nếu scheduler cần khác tĩnh
            pass

        # In classification report để debug (mỗi 2 epoch)
        if epoch % 2 == 0:
            print("📊 Classification report (validation):")
            try:
                if class_names is not None:
                    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
                else:
                    print(classification_report(all_labels, all_preds, zero_division=0))
            except Exception as e:
                print("⚠️ Không thể in classification_report:", e)

        # Early stopping đơn giản
        if no_improve >= patience:
            print(f"⛔ Early stopping: không cải thiện sau {patience} epoch.")
            break

    print("\n✅ Huấn luyện hoàn tất!")
