import matplotlib as plt
import os
def draw(valLossHistory,savePath):
    # === 训练完成后，画出 loss 曲线 ===
    plt.figure(figsize=(8, 6))
    plt.plot(valLossHistory, label='Validation Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 保存图像
    plt_path = os.path.join(savePath, 'val_loss_curve.png')
    plt.savefig(plt_path)
    print(f"[INFO] Validation loss curve saved to {plt_path}")
    plt.close()