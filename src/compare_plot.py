def plot_comparison(model_names, accuracies, save_path='model_comparison.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    x_pos = np.arange(len(model_names))
    
    plt.bar(x_pos, accuracies, align='center', alpha=0.7, color=['blue', 'orange', 'green'])
    plt.xticks(x_pos, model_names)
    plt.ylabel('Accuracy')
    plt.title('Model Comparison on Telco Customer Churn Dataset')

    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"模型对比图已保存到 {save_path}")
    plt.close()