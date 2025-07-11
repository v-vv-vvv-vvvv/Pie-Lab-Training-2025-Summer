import matplotlib.pyplot as plt

test_accs = [0.5109, 0.6916, 0.7363, 0.7826, 0.8194, 0.8165, 0.8239, 0.8301, 0.8405, 0.8362, 0.8376, 0.8388, 0.8373, 0.8467, 0.844, 0.839, 0.8381, 0.8397, 0.8511, 0.847]
train_losses = [1.4100460982536112, 0.9079202971495021, 0.6763846839556609, 0.5300840716570845, 0.4273569495857829, 0.34663080775638677, 0.2704878064906201, 0.20705181054408897, 0.15736040809784857, 0.11768676339627226, 0.09094274048919758, 0.07741985560211417, 0.0642607615334089, 0.057788166332283696, 0.051712787745412096, 0.046988212850803505, 0.044017647689966424, 0.04062325102114833, 0.038132948260825805, 0.03172374858877912]


tab = [0.2742, 0.3569, 0.385, 0.5133, 0.5861, 0.6344, 0.6775, 0.6591, 0.6604, 0.6969, 0.74, 0.7386, 0.7683, 0.7919, 0.7922, 0.7561, 0.7903, 0.8216, 0.8187, 0.8269]
tlb = [2.0055643856677863, 1.7851792694357655, 1.5843144763461159, 1.3329643615523872, 1.1420547278488384, 1.0294832173363326, 0.9348779865695388, 0.8646458873663412, 0.7968532204475549, 0.7370135457543157, 0.6810242149149975, 0.6203650970898016, 0.5762655473578616, 0.5308918048963522, 0.49319504071836884, 0.4561919616844953, 0.41938220345608107, 0.38949534173130684, 0.35827033800999525, 0.3274641605117894]

# 绘制训练曲线
plt.figure(figsize=(7, 5))

# 损失曲线
plt.subplot(1, 1, 1)
plt.plot(train_losses, label='Res Train Loss', color='blue', linewidth=2)
plt.plot(tlb, label='Train Loss', color='blue', linewidth=1)
plt.plot(test_accs, label='Res Test Accuracy', color='red', linewidth=2)
plt.plot(tab, label='Test Accuracy', color='red', linewidth=1)
plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Loss', fontsize=12)
# plt.title('Training and Test Loss', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

# # 准确率曲线
# plt.subplot(1, 2, 2)
# plt.plot(train_accs, label='Train Accuracy', color='blue', linewidth=2)
# plt.plot(test_accs, label='Test Accuracy', color='red', linewidth=2)
# print(test_accs)
# print(train_losses)
# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Accuracy', fontsize=12)
# plt.title('Training and Test Accuracy', fontsize=14)
# plt.legend(fontsize=10)
# plt.grid(True, linestyle='--', alpha=0.5)

# 调整布局并保存
plt.tight_layout()
plt.savefig(f'34_curves.png', dpi=300, bbox_inches='tight')