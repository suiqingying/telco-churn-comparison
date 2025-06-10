from src.data_loader import load_data
from src.logistic_regression import LogisticRegressionModel
from src.decision_tree import DecisionTreeModel
from src.boosting import BoostingModel
from src.compare_plot import plot_comparison
import matplotlib.pyplot as plt

import time
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

def plot_roc_curve(y_true, y_score, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()

    logistic_model = LogisticRegressionModel()
    decision_tree_model = DecisionTreeModel()
    boosting_model = BoostingModel()

    results = []
    report_lines = []

    # Logistic Regression
    report_lines.append("=== Logistic Regression ===")
    start = time.time()
    logistic_model.train(X_train, y_train)
    train_time = time.time() - start
    y_pred = logistic_model.model.predict(X_test)
    y_score = logistic_model.model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)
    acc = accuracy_score(y_test, y_pred)
    report_lines.append(f"训练时间: {train_time:.2f} 秒")
    report_lines.append(f"准确率: {acc:.4f}")
    report_lines.append(f"AUC: {auc:.4f}")
    report_lines.append("分类报告:\n" + classification_report(y_test, y_pred))
    report_lines.append("混淆矩阵:\n" + str(confusion_matrix(y_test, y_pred)))
    plot_roc_curve(y_test, y_score, "Logistic Regression")
    results.append(('Logistic Regression', acc))

    # Decision Tree
    report_lines.append("\n=== Decision Tree ===")
    start = time.time()
    decision_tree_model.train(X_train, y_train)
    train_time = time.time() - start
    y_pred = decision_tree_model.model.predict(X_test)
    if hasattr(decision_tree_model.model, "predict_proba"):
        y_score = decision_tree_model.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_score)
    else:
        y_score = None
        auc = None
    acc = accuracy_score(y_test, y_pred)
    report_lines.append(f"训练时间: {train_time:.2f} 秒")
    report_lines.append(f"准确率: {acc:.4f}")
    report_lines.append(f"AUC: {auc:.4f}")
    report_lines.append("分类报告:\n" + classification_report(y_test, y_pred))
    report_lines.append("混淆矩阵:\n" + str(confusion_matrix(y_test, y_pred)))
    if y_score is not None:
        plot_roc_curve(y_test, y_score, "Decision Tree")
    results.append(('Decision Tree', acc))

    # Boosting
    report_lines.append("\n=== Boosting ===")
    start = time.time()
    boosting_model.train(X_train, y_train)
    train_time = time.time() - start
    y_pred = boosting_model.model.predict(X_test)
    if hasattr(boosting_model.model, "predict_proba"):
        y_score = boosting_model.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_score)
    else:
        y_score = None
        auc = None
    acc = accuracy_score(y_test, y_pred)
    report_lines.append(f"训练时间: {train_time:.2f} 秒")
    report_lines.append(f"准确率: {acc:.4f}")
    report_lines.append(f"AUC: {auc:.4f}")
    report_lines.append("分类报告:\n" + classification_report(y_test, y_pred))
    report_lines.append("混淆矩阵:\n" + str(confusion_matrix(y_test, y_pred)))
    if y_score is not None:
        plot_roc_curve(y_test, y_score, "Boosting")
    results.append(('Boosting', acc))

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
    plot_comparison(model_names, accuracies, save_path='fig/model_comparison.png')

if __name__ == "__main__":
    main()