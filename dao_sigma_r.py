import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings
from datetime import datetime

# 忽略数值计算中可能产生的除零警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================================
# ★ 关键参数配置区 (Dao et al. 量纲分析模型)
# ==========================================================
# 公式: hp/hmax = C1*(ln(sigma_r/Er))^3 + C2*(ln(sigma_r/Er))^2 + C3*ln(sigma_r/Er) + C4
C1 = -0.0041
C2 = -0.0882
C3 = -0.6542
C4 = -0.6729

# 配置需要参与 Dao 模型计算的弹性模量列 (仅保留 Er 和 Er_O&P)
E_COLUMNS_TO_PROCESS = ['Er', 'Er_O&P']


def dao_stress_equation(sigma_r, hp_hmax, Er):
    """
    根据 Dao 量纲分析模型构造的隐式方程：
    f(sigma_r) = c1*(ln(sigma_r/Er))^3 + c2*(ln(sigma_r/Er))^2 + c3*ln(sigma_r/Er) + c4 - (hp/hmax) = 0
    """
    # 物理防线：应力不能为负或零
    if sigma_r <= 0:
        return 1e9

    # 核心计算：Dao 模型中的对数项为 ln(sigma_r / Er)，注意与 Yao 模型的分子分母是反过来的！
    ln_term = np.log(sigma_r / Er)

    # 计算公式右侧多项式
    right_side = C1 * (ln_term ** 3) + C2 * (ln_term ** 2) + C3 * ln_term + C4

    # 返回差值，供 fsolve 寻找零点（当返回值为 0 时，即 right_side == hp_hmax）
    return right_side - hp_hmax


def calculate_dao_sigma_r():
    print("=" * 70)
    print("🚀 启动 Dao 量纲分析模型 - 表征应力多模量并行求解引擎")
    print("=" * 70)

    # ==========================================================
    # 1. 目录架构配置与时间戳生成
    # ==========================================================
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_dir = r"D:\Woooooork\berkovich_C\data"
    # 输出到专门的 dao_sigma_r 文件夹下
    base_output_dir = r"D:\Woooooork\berkovich_C\output\dao_sigma_r"

    run_output_dir = os.path.join(base_output_dir, f"Run_{time_str}")
    os.makedirs(run_output_dir, exist_ok=True)

    known_data_path = os.path.join(data_dir, "knowndata.csv")
    output_csv_path = os.path.join(run_output_dir, "Calculated_Sigma_r_Dao_Results.csv")
    plot_save_path = os.path.join(run_output_dir, "Yield_Strength_Comparison_Dao.png")

    # ==========================================================
    # 2. 稳健读取数据 (Dao 方法不需要依赖 C 值汇总表)
    # ==========================================================
    encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'ansi', 'latin1']
    df_known = None
    for enc in encodings_to_try:
        try:
            df_known = pd.read_csv(known_data_path, encoding=enc)
            print(f"✅ 成功使用 '{enc}' 编码读取 known_data 表格！")
            break
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            print(f"【错误】找不到文件 {known_data_path}。")
            return

    if df_known is None:
        print("【致命错误】所有编码均无法解析 knowndata.csv。")
        return

    try:
        # 清理表头隐藏空格
        df_known.columns = df_known.columns.str.strip()
        df_known['Material'] = df_known['Material'].astype(str).str.strip()
    except Exception as e:
        print(f"【错误】清理表头数据失败: {e}")
        return

    # 校验必要列是否存在
    if 'hp_hmax' not in df_known.columns:
        print("【错误】在 CSV 中找不到关键列 'hp_hmax'，Dao 模型无法计算！")
        return

    actual_e_cols = [col for col in E_COLUMNS_TO_PROCESS if col in df_known.columns]
    if not actual_e_cols:
        print(f"【错误】在 CSV 中找不到您配置的任何 E 列: {E_COLUMNS_TO_PROCESS}")
        print(f"当前 CSV 拥有的列名为: {list(df_known.columns)}")
        return

    has_yield_stress = 'sigma_y_m' in df_known.columns

    print(f"✅ 准备就绪，共载入 {len(df_known)} 种材料数据。即将并行求解 {len(actual_e_cols)} 种模量...")
    print("-" * 70)

    # ==========================================================
    # 3. 遍历求解表征应力 (多模量并行版)
    # ==========================================================
    for e_col in actual_e_cols:
        df_known[f'Sigma_r_from_{e_col}'] = np.nan

    for index, row in df_known.iterrows():
        material = row['Material']
        hp_hmax = row['hp_hmax']

        # 如果 hp_hmax 数据缺失或不合法，直接跳过
        if pd.isna(hp_hmax) or hp_hmax <= 0:
            print(f"👉 {material: <20} | [hp_hmax 异常跳过]")
            continue

        print(f"👉 {material: <20} | hp/hmax={hp_hmax:.3f} | ", end="")

        for e_col in actual_e_cols:
            Er = row[e_col]

            # 容错：跳过模量空值或非正数
            if pd.isna(Er) or Er <= 0:
                print(f"[{e_col}: 缺失] ".ljust(15), end="")
                continue

            # 初始猜测值：表征应力通常比弹性模量小 1~2 个数量级
            initial_guess = Er / 50.0

            try:
                root, infodict, ier, mesg = fsolve(
                    dao_stress_equation,
                    x0=initial_guess,
                    args=(hp_hmax, Er),
                    full_output=True
                )

                if ier == 1:
                    sigma_r_fit = root[0]
                    df_known.at[index, f'Sigma_r_from_{e_col}'] = sigma_r_fit
                    print(f"[{e_col}]: {sigma_r_fit:.2f} ".ljust(15), end="")
                else:
                    print(f"[{e_col}: 未收敛] ".ljust(15), end="")

            except Exception:
                print(f"[{e_col}: 异常] ".ljust(15), end="")

        print()

    # ==========================================================
    # 4. 生成交叉验证散点图 (sigma_y_m vs calculated sigma_r)
    # ==========================================================
    if has_yield_stress:
        print("-" * 70)
        print("📊 正在绘制 Dao 模型屈服强度交叉验证拼图...")

        # 两种 E，所以直接画成 1行 x 2列
        n_cols = 2
        n_rows = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5))

        # 保证 axes 是可迭代对象
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for i, e_col in enumerate(actual_e_cols):
            ax = axes[i]
            y_col = f'Sigma_r_from_{e_col}'

            # 剥离缺失值
            plot_df = df_known[['sigma_y_m', y_col]].dropna()

            if not plot_df.empty:
                # 散点绘制
                ax.scatter(plot_df['sigma_y_m'], plot_df[y_col], color='teal', edgecolor='black', alpha=0.7, s=50,
                           label='Dao Model vs Reference')

                # 动态获取 y=x 虚线范围
                min_val = min(plot_df['sigma_y_m'].min(), plot_df[y_col].min()) * 0.9
                max_val = max(plot_df['sigma_y_m'].max(), plot_df[y_col].max()) * 1.1

                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6, linewidth=2,
                        label='y = x (Perfect Fit)')

                # 图表装潢
                ax.set_title(f'Dao Model - Using Modulus: {e_col}', fontsize=11, fontweight='bold')
                ax.set_xlabel(r'Reference Yield Strength $\sigma_{y,m}$ (GPa)', fontsize=10)
                ax.set_ylabel(r'Calculated $\sigma_r$ (GPa)', fontsize=10)
                ax.legend(loc='upper left')
                ax.grid(True, linestyle=':', alpha=0.6)
            else:
                ax.set_title(f'No valid data for {e_col}')
                ax.axis('off')

        # 隐藏多余子图 (虽然当前配置了2个刚刚好，留作扩展防错)
        for j in range(len(actual_e_cols), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(plot_save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✨ 验证对比图已生成: {plot_save_path}")

    # ==========================================================
    # 5. 数据列整理与安全导出
    # ==========================================================
    output_columns = ['Material', 'hp_hmax']
    if has_yield_stress:
        output_columns.append('sigma_y_m')

    for e_col in actual_e_cols:
        output_columns.append(e_col)
        output_columns.append(f'Sigma_r_from_{e_col}')

    all_cols = [c for c in df_known.columns if c not in output_columns]
    df_final = df_known[output_columns + all_cols]

    try:
        df_final.to_csv(output_csv_path, index=False, float_format="%.4f", encoding='utf-8-sig')
        print("-" * 70)
        print(f"🎉 Dao 模型并行计算全部完成！")
        print(f"📁 结果与图像已安全存入专属档案库:\n   {run_output_dir}")
        print("=" * 70)
    except PermissionError:
        print("【错误】导出失败！CSV 文件正在被占用，请关闭 Excel 后再试。")
    except Exception as e:
        print(f"【错误】导出 CSV 失败: {e}")


if __name__ == "__main__":
    calculate_dao_sigma_r()