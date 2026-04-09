"""
=============================================================================================
符号回归纳米压入本构方程探索引擎 (Symbolic Regression Constitutive Equation Explorer)
=============================================================================================

【功能概述】
本项目利用基于遗传编程的符号回归算法（PySR），在无需预设数学模型的前提下，
从纳米压入试验数据中自动搜寻并推导表征应力/屈服强度的最佳经验方程。

【主要工作流程】
1. 数据预处理：从本地 CSV 读取数据，清洗缺失值，分离输入特征与目标变量。
2. 符号回归训练：配置遗传算法参数（种群、惩罚系数、允许的算子），通过迭代搜寻低损失、高简洁度的数学方程。
3. 符号表达式解析：利用 SymPy 库解析最佳方程，并将其内部的浮点常数严格截断至两位小数以增强可读性。
4. 动态可视化：
   - 图表 1 (Parity Plot)：绘制预测值与参考值的对比散点图，直观评估拟合精度。
   - 图表 2 (Function Plot)：动态检测最佳方程所包含的特征维度，自动生成 2D 物理曲线或 3D 物理曲面。
   - 图表 3 (AST Tree)：利用原生 Matplotlib 递归渲染方程的抽象语法树，解析方程的底层逻辑结构。

【核心输入与输出】
- 输入：`knowndata.csv`，需包含特征列（如 'hp_hmax', 'Er'）及目标验证列（'sigma_y_m'）。
- 输出：带时间戳的专属结果文件夹，内含备选公式列表 (CSV) 及三张高保真可视化评估图片。

【关键运行条件】
- 依赖库：numpy, pandas, matplotlib, pysr, sympy
- 环境变量：PySR 强依赖底层的 Julia 引擎，首次运行需确保 Julia 环境及相关包管理器已正确配置。
=============================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pysr import PySRRegressor
import sympy
from sympy import lambdify
import warnings
from datetime import datetime

# 忽略数值计算及 Pandas 运行过程中的常规警告，保持控制台输出整洁
warnings.filterwarnings("ignore")


# ==========================================================
# ★ 纯 Python 实现的 AST 树状图递归绘制引擎
# ==========================================================
def plot_sympy_tree(expr, ax, x=0.5, y=1.0, dx=0.25, dy=0.15):
    """
    不依赖 Graphviz 等外部渲染引擎，利用原生 matplotlib 递归遍历并绘制 SymPy 表达式树。

    参数说明:
    - expr: 当前的 SymPy 表达式节点
    - ax: matplotlib 的 Axes 对象，用于承载图形
    - x, y: 当前节点的绘制中心坐标
    - dx, dy: 子节点在横向和纵向的分布步长，随递归深度动态递减以防止节点重叠
    """
    if not expr.args:
        # 边界条件：若无子参数，说明到达叶子节点（如具体的特征变量 'Er' 或常数数字）
        node_text = str(expr)
        ax.text(x, y, node_text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle="circle,pad=0.4", fc="honeydew", ec="gray"), zorder=3)
        return

    # 获取当前操作符的内部类型名称（如 Add, Mul, Pow）
    func_name = type(expr).__name__

    # 将面向计算机的类型映射为人类直觉的数学符号
    if func_name == 'Add':
        func_name = '+'
    elif func_name == 'Mul':
        func_name = '×'
    elif func_name == 'Pow':
        func_name = '^'

    # 绘制内部节点（算子）
    ax.text(x, y, func_name, ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", fc="aliceblue", ec="gray"), zorder=3)

    # 获取该算子包含的子节点数量，并计算其对称分布的起始 X 坐标
    n_children = len(expr.args)
    if n_children > 0:
        start_x = x - dx * (n_children - 1) / 2

        # 遍历所有子节点，绘制连线并下探递归
        for i, arg in enumerate(expr.args):
            child_x = start_x + i * dx
            child_y = y - dy

            # 绘制父节点到子节点的连接线（zorder设为1，保证线条在文本框下方）
            ax.plot([x, child_x], [y, child_y], 'k-', lw=1.5, zorder=1, alpha=0.6)

            # 递归调用，dx 除以 1.6 以收敛横向空间，防止树状图下层过于宽泛导致越界
            plot_sympy_tree(arg, ax, child_x, child_y, dx / 1.6, dy)


def run_symbolic_regression():
    print("=" * 70)
    print("🧠 启动 PySR 符号回归引擎：探索未知的压入本构方程")
    print("=" * 70)

    # ==========================================================
    # 1. 读取与清洗数据
    # ==========================================================
    data_dir = r"D:\Woooooork\berkovich_C\data"
    known_data_path = os.path.join(data_dir, "knowndata.csv")

    try:
        # 指定 gbk 编码以兼容中文或 Windows Excel 导出的 CSV 文件
        df = pd.read_csv(known_data_path, encoding='gbk')
        # 去除表头可能存在的隐藏空格，避免后续索引报错
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"【错误】读取数据失败: {e}")
        return

    # 指定符号回归算法尝试使用的底层物理特征组合
    # 修改这里的列表，即可改变参与构建的特征参数
    features = ['hp_hmax', 'Er', 'E*']
    # 目标变量（即等号左边的 Y 值）
    target = 'sigma_y_m'

    if target not in df.columns:
        print(f"【错误】找不到目标列 '{target}'。")
        return

    # 剔除在指定特征和目标列中存在缺失值的记录，保证训练数据的完整性（加入 .copy() 防止切片警告）
    df_clean = df.dropna(subset=features + [target]).copy()

    # ==========================================================
    # 【核心修复】PySR 变量名合法化清洗
    # PySR 底层引擎会将特征名作为代码变量编译，严禁包含 '*'、'/' 等符号
    # 我们在内存中将其转换为安全的标识符，例如 'E*' -> 'E_star'
    # ==========================================================
    rename_dict = {}
    for col in features:
        safe_col = col.replace('*', '_star').replace('/', '_div_').replace('-', '_')
        rename_dict[col] = safe_col

    df_clean.rename(columns=rename_dict, inplace=True)
    # 同步更新特征列表，告诉 AI 使用清洗后的安全名称
    features = [rename_dict[f] for f in features]

    # 物理防线：样本量过少容易导致遗传算法寻找出严重过拟合的畸形长公式
    if len(df_clean) < 10:
        print("【警告】有效数据点太少，符号回归容易过拟合。建议拥有更多数据！")

    X = df_clean[features]
    y = df_clean[target]

    print(f"✅ 成功载入 {len(df_clean)} 组有效数据。")
    print(f"   -> 输入特征 (X): {features}")
    print(f"   -> 目标变量 (y): {target}")

    # ==========================================================
    # 2. 构建带时间戳的专属输出目录
    # ==========================================================
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    features_str = "_".join(features)

    # 此时 features_str 已经通过前一步清洗去除了特殊字符，可直接作为安全的文件夹名称
    folder_name = f"{features_str}_to_{target}_{time_str}"

    base_output_dir = r"D:\Woooooork\berkovich_C\output\sym_reg"
    run_output_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(run_output_dir, exist_ok=True)

    # 设定回归方程集的保存路径，避免在项目根目录产生凌乱的缓存文件
    eq_file_path = os.path.join(run_output_dir, "PySR_Equations.csv")

    print("-" * 70)
    print("⏳ 正在启动符号回归进化算法，这可能需要几分钟...")

    # ==========================================================
    # 3. 配置并训练 PySR 符号回归模型
    # ==========================================================
    model = PySRRegressor(
        # 允许使用的基础数学符号池，涵盖了主流本构模型需要的指数与对数运算
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log"],
        # niterations: 遗传编程的繁衍代数。40 为兼顾速度与初步探索的适中值
        niterations=100,
        populations=15,
        # parsimony (简约惩罚系数): 值越大，AI 越倾向于舍弃微小的精度提升来换取公式的简短。
        # 鉴于压入数据样本较少，设为 0.01 强迫模型给出具备物理泛化能力的精简公式
        parsimony=0.01,
        equation_file=eq_file_path
    )

    # 启动多线程/多进程进化寻优过程
    model.fit(X, y)

    # ==========================================================
    # 4. 获取并格式化最佳公式 (保留两位小数)
    # ==========================================================
    best_equation_sympy = model.sympy()

    # 精准控制浮点数精度：利用 xreplace 遍历语法树的所有原子节点 (atoms)，
    # 仅将其中的浮点数 (Float) 四舍五入保留两位小数，不改变原有公式的数学代数结构
    rounded_equation_sympy = best_equation_sympy.xreplace(
        {n: round(n, 2) for n in best_equation_sympy.atoms(sympy.Float)}
    )
    best_equation_str = str(rounded_equation_sympy)

    # 生成预测值，为后续拟合评估绘图准备数据
    y_pred = model.predict(X)

    # ==========================================================
    # 5. 绘制图表 1：预测精度图 (Parity Plot: y=x)
    # ==========================================================
    print("\n📊 正在生成拟合精度评估图 (Parity Plot)...")
    fig1, ax1 = plt.subplots(figsize=(8, 7))

    # 调整底部边界，为全局公式文本框预留 25% 的垂直空间
    plt.subplots_adjust(bottom=0.25)

    ax1.scatter(y, y_pred, color='coral', edgecolor='black', alpha=0.8, s=60, label='PySR Predictions')

    # 根据真实值和预测值动态计算 y=x 参考线的合理延伸范围
    min_val = min(y.min(), y_pred.min()) * 0.9
    max_val = max(y.max(), y_pred.max()) * 1.1
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6, linewidth=2, label='y = x (Perfect Fit)')

    ax1.set_title('Model Accuracy: Predicted vs Reference', fontsize=12, fontweight='bold')
    ax1.set_xlabel(r'Reference Yield Strength $\sigma_{y,m}$ (GPa)', fontsize=11)
    ax1.set_ylabel(r'Predicted Yield Strength $\sigma_y$ (GPa)', fontsize=11)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 将格式化后的最终方程固定悬浮于画布底部中央
    fig1.text(0.5, 0.08, f"Discovered Equation:\n$\sigma_y$ = {best_equation_str}",
              ha='center', va='center', fontsize=12,
              bbox=dict(boxstyle="round,pad=0.6", edgecolor='gray', facecolor='aliceblue', alpha=0.9))

    fig1.savefig(os.path.join(run_output_dir, "1_Accuracy_Parity_Plot.png"), dpi=200)
    plt.close(fig1)

    # ==========================================================
    # 6. 绘制图表 2：物理方程曲线/曲面图 (Function Curve/Surface)
    # ==========================================================
    print("📈 正在生成物理方程曲线图 (Equation Plot)...")

    # 动态探测 AI 最终选用的变量维度：提取解析式中的自由符号 (如仅使用 Er，或同时使用 Er, hp_hmax)
    free_symbols = list(rounded_equation_sympy.free_symbols)
    active_features = [str(sym) for sym in free_symbols]

    # lambdify: 将缓慢的符号运算转换为支持 Numpy 矩阵化高速计算的函数映射
    func = lambdify(free_symbols, rounded_equation_sympy, modules=['numpy'])

    fig2 = plt.figure(figsize=(9, 8))

    if len(active_features) == 1:
        # 【场景 A：1D 依赖】仅提取到一个关键特征，生成二维曲线图
        ax2 = fig2.add_subplot(111)
        feat_name = active_features[0]

        # 映射原始散点数据
        ax2.scatter(df_clean[feat_name], y, color='teal', edgecolor='black', alpha=0.8, s=60, label='Raw Data')

        # 构建高密度等距数组并计算平滑响应曲线
        x_vals = np.linspace(df_clean[feat_name].min(), df_clean[feat_name].max(), 200)
        y_vals = func(x_vals)
        ax2.plot(x_vals, y_vals, color='red', linewidth=2.5, label=f'Equation Curve')

        ax2.set_xlabel(feat_name, fontsize=12)
        ax2.set_ylabel(r'Yield Strength $\sigma_y$ (GPa)', fontsize=12)
        ax2.set_title('Discovered Physical Function Curve', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.subplots_adjust(bottom=0.25)
        fig2.text(0.5, 0.08, f"Equation:\n$\sigma_y$ = {best_equation_str}",
                  ha='center', va='center', fontsize=12,
                  bbox=dict(boxstyle="round,pad=0.6", edgecolor='gray', facecolor='honeydew', alpha=0.9))

    elif len(active_features) == 2:
        # 【场景 B：2D 依赖】提取到两个协同特征，生成三维连续曲面图
        ax2 = fig2.add_subplot(111, projection='3d')
        f1, f2 = active_features[0], active_features[1]

        ax2.scatter(df_clean[f1], df_clean[f2], y, color='teal', edgecolor='black', alpha=1.0, s=40, label='Raw Data')

        # 在特征空间内生成二维网格平面矩阵，代入方程运算出三维曲面 Z 坐标
        x_vals = np.linspace(df_clean[f1].min(), df_clean[f1].max(), 30)
        y_vals = np.linspace(df_clean[f2].min(), df_clean[f2].max(), 30)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        Z_grid = func(X_grid, Y_grid)

        ax2.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.6, edgecolor='none')

        ax2.set_xlabel(f1, fontsize=10)
        ax2.set_ylabel(f2, fontsize=10)
        ax2.set_zlabel(r'$\sigma_y$ (GPa)', fontsize=10)
        ax2.set_title('Discovered Physical Function Surface', fontsize=13, fontweight='bold')

        # 对于 3D 绘图，因视角问题，公式适宜悬浮标注于画布正上方
        fig2.text(0.5, 0.95, f"Equation: $\sigma_y$ = {best_equation_str}",
                  ha='center', va='top', fontsize=12,
                  bbox=dict(boxstyle="round,pad=0.4", edgecolor='gray', facecolor='honeydew', alpha=0.9))
    else:
        # 场景 C：回退处理。当回归结果退化为常数，或特征过多（>2）无法直观降维时，提供文字提示
        plt.text(0.5, 0.5, "Equation is a constant or has too many features to plot.", ha='center')

    fig2.savefig(os.path.join(run_output_dir, "2_Physical_Equation_Plot.png"), dpi=200)
    plt.close(fig2)

    # ==========================================================
    # 7. 绘制图表 3：抽象语法树 (纯 Python AST)
    # ==========================================================
    print("🌳 正在生成抽象语法树图 (AST)...")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.axis('off')  # 完全隐藏坐标轴外框和刻度

    # 动态布局逻辑：获取所有基础元素的数量，若公式过于臃肿（>10 个节点），
    # 则相应扩大画布尺寸并增加起始横向步长，以防止复杂子树的文本重叠
    initial_dx = 0.35
    if len(rounded_equation_sympy.atoms()) > 10:
        initial_dx = 0.5
        fig3.set_size_inches(14, 9)

    # 调用顶层定义的纯 Python 渲染算法启动递归绘制
    plot_sympy_tree(rounded_equation_sympy, ax3, x=0.5, y=1.0, dx=initial_dx, dy=0.15)

    ax3.set_title("Abstract Syntax Tree (AST)", fontsize=16, fontweight='bold', pad=20)

    # tight 模式可自动裁剪画布边缘留白
    tree_save_path = os.path.join(run_output_dir, "3_Equation_AST_Tree.png")
    fig3.savefig(tree_save_path, dpi=200, bbox_inches='tight')
    plt.close(fig3)

    # ==========================================================
    # 8. 最终信息打印
    # ==========================================================
    print("\n" + "=" * 70)
    print("🎉 符号回归探索与全套动态绘图全部完成！")
    print("=" * 70)
    print("\n🏆 AI 发现的最佳经验公式 (保留两位小数)：")
    print(best_equation_str)
    print(f"\n📁 所有的公式列表及图表 (1_Parity, 2_Curve/Surface, 3_AST_Tree) 已安全存放至:\n   {run_output_dir}")


if __name__ == "__main__":
    run_symbolic_regression()