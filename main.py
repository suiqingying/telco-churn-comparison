from src.utils.data_loader import load_data
from src.linear_methods.logistic_regression import LogisticRegressionModel
from src.linear_methods.linear_svm import LinearSVMModel
from src.nonlinear_methods.kernel_svm import KernelSVMModel
from src.nonlinear_methods.decision_tree import DecisionTreeModel
from src.ensemble_methods.bagging import BaggingModel
from src.ensemble_methods.boosting import BoostingModel
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.tree import export_graphviz
import graphviz

def visualize_decision_tree(model, feature_names, class_names, output_file="fig/decision_tree.png"):
    # 导出决策树为 DOT 格式
    dot_data = export_graphviz(
        model.model,  # 假设模型内部有一个 `model` 属性存储 sklearn 的决策树
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    # 使用 graphviz 绘制决策树
    graph = graphviz.Source(dot_data)
    graph.render(output_file, format="png", cleanup=True)
    print(f"决策树已保存到 {output_file}")

def plot_roc_curve(y_true, y_score, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()

    models = [
        ("Logistic Regression", LogisticRegressionModel()),
        ("Linear SVM", LinearSVMModel()),
        ("Kernel SVM", KernelSVMModel()),
        ("Decision Tree", DecisionTreeModel()),
        ("Bagging", BaggingModel()),
        ("Boosting", BoostingModel())
    ]

    results = []
    report_lines = []

    for model_name, model in models:
        report_lines.append(f"\n=== {model_name} ===")
        start = time.time()
        model.train(X_train, y_train)
        train_time = time.time() - start
        y_pred = model.predict(X_test)
        y_score = model.model.predict_proba(X_test)[:, 1] if hasattr(model.model, "predict_proba") else None
        auc = roc_auc_score(y_test, y_score) if y_score is not None else "N/A"
        acc = accuracy_score(y_test, y_pred)
        report_lines.append(f"训练时间: {train_time:.2f} 秒")
        report_lines.append(f"准确率: {acc:.4f}")
        report_lines.append(f"AUC: {auc:.4f}" if auc != "N/A" else "AUC: 不支持")
        report_lines.append("分类报告:\n" + classification_report(y_test, y_pred))
        report_lines.append("混淆矩阵:\n" + str(confusion_matrix(y_test, y_pred)))
        if y_score is not None:
            plot_roc_curve(y_test, y_score, model_name)
        results.append((model_name, acc, train_time))
        if model_name == "Decision Tree":
            feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]
            class_names = ["No Churn", "Churn"]
            visualize_decision_tree(model, feature_names, class_names, output_file="fig/decision_tree")

    # 保存所有结果到 result.txt
    with open("results/result.txt", "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")
    print("所有模型结果已保存到 result.txt")

    # 绘制ROC曲线
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig/roc_curve.png')
    plt.close()

    # 绘制对比图
    model_names = [r[0] for r in results]
    accuracies = [r[1] for r in results]
    train_times = [r[2] for r in results]
    
    # 绘制准确率对比图
    plt.figure(figsize=(10, 6))  # 增加图像尺寸
    plt.bar(model_names, accuracies, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.xticks(rotation=45)  # 调整横坐标标签角度
    plt.tight_layout()
    plt.savefig('fig/model_comparison_accuracy.png')
    plt.close()

    # 绘制训练时间对比图
    plt.figure(figsize=(10, 6))  # 增加图像尺寸
    plt.bar(model_names, train_times, color='orange')
    plt.xticks(rotation=45)  # 调整横坐标标签角度
    plt.xlabel('Model')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.tight_layout()
    plt.savefig('fig/model_comparison_training_time.png')
    plt.close()

if __name__ == "__main__":
    main()