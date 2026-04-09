import os
import glob
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
# ★ 关键参数配置区
# ==========================================================
# 默认填入的是论文中 60° 压头的值。
# 【强烈建议】：请务必查阅您的 image_4adf00.png 截图，
# 将下面四个常数修改为 Berkovich 压头（等效 70.3°）对应的值！
C1 = -1.131  # 对应 ln 的 3 次方系数
C2 = 13.635  # 对应 ln 的 2 次方系数
C3 = -30.594  # 对应 ln 的 1 次方系数
C4 = 29.267  # 常数项

# 【新增配置】：在这里填入 knowndata.csv 中 4 种 E 的准确列名
E_COLUMNS_TO_PROCESS = ['Er', 'Er_O&P', 'E*', 'E*_O&P']


def representative_stress_equation(sigma_r, C, Er):
    """
    根据姚博论文 CR-EMI 方法公式 (3) 构造的隐式方程：
    C / sigma_r = c1*(ln(Er/sigma_r))^3 + c2*(ln(Er/sigma_r))^2 + c3*ln(Er/sigma_r) + c4
    """
    if sigma_r <= 0:
        return 1e9

    ln_term = np.log(Er / sigma_r)
    right_side = C1 * (ln_term ** 3) + C2 * (ln_term ** 2) + C3 * ln_term + C4

    return right_side - (C / sigma_r)


def calculate_yield_stress():
    print("=" * 70)
    print("🚀 启动 CR-EMI 屈服强度 (n=0) 多模量并行验证引擎")
    print("=" * 70)

    # ==========================================================
    # 1. 目录架构配置与时间戳生成
    # ==========================================================
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_dir = r"D:\Woooooork\berkovich_C\data"
    base_output_dir = r"D:\Woooooork\berkovich_C\output\yao_sigma_r"
    yao_c_output_dir = r"D:\Woooooork\berkovich_C\output\yao_C"

    run_output_dir = os.path.join(base_output_dir, f"Run_{time_str}")
    os.makedirs(run_output_dir, exist_ok=True)

    known_data_path = os.path.join(data_dir, "knowndata.csv")
    output_csv_path = os.path.join(run_output_dir, "Calculated_Sigma_r_Multi_E_Results.csv")
    plot_save_path = os.path.join(run_output_dir, "Yield_Strength_Comparison_Grid.png")

    # ==========================================================
    # 2. 智能检索最新生成的 C 值汇总表
    # ==========================================================
    run_folders = glob.glob(os.path.join(yao_c_output_dir, "Run_*"))
    if not run_folders:
        print(f"【错误】在 {yao_c_output_dir} 下未找到 C 值计算记录。请先运行 calculate_C 程序。")
        return

    latest_run_dir = max(run_folders, key=os.path.getctime)
    c_summary_path = os.path.join(latest_run_dir, "Global_Stable_C_Summary.txt")

    if not os.path.exists(c_summary_path):
        print(f"【错误】在最新记录 {latest_run_dir} 中未找到 C 值汇总文件。")
        return

    print(f"🔍 智能检索：锁定最新 C 值汇总表 -> {os.path.basename(latest_run_dir)}")

    # ==========================================================
    # 3. 稳健读取数据
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
        df_known.columns = df_known.columns.str.strip()
        df_known['Material'] = df_known['Material'].astype(str).str.strip()

        df_c = pd.read_csv(c_summary_path, sep='\t')
        df_c.columns = df_c.columns.str.strip()
        df_c['Material_Name'] = df_c['Material_Name'].astype(str).str.strip()

    except Exception as e:
        print(f"【错误】清理表头数据失败: {e}")
        return

    # 校验用户配置的 E 列是否存在于 CSV 中
    actual_e_cols = [col for col in E_COLUMNS_TO_PROCESS if col in df_known.columns]
    if not actual_e_cols:
        print(f"【错误】在 CSV 中找不到您配置的任何 E 列: {E_COLUMNS_TO_PROCESS}")
        print(f"当前 CSV 拥有的列名为: {list(df_known.columns)}")
        return

    # 检查是否存在屈服强度验证列
    has_yield_stress = 'sigma_y_m' in df_known.columns

    # ==========================================================
    # 4. 自动化数据融合 (Merge)
    # ==========================================================
    df_merged = pd.merge(df_known, df_c, left_on='Material', right_on='Material_Name', how='inner')

    if df_merged.empty:
        print("【错误】数据合并失败！材料名称不匹配。")
        return

    print(f"✅ 成功融合 {len(df_merged)} 种材料的数据。即将并行求解 {len(actual_e_cols)} 种模量...")
    print("-" * 70)

    # ==========================================================
    # 5. 遍历求解表征应力 (多模量并行版)
    # ==========================================================
    for e_col in actual_e_cols:
        df_merged[f'Sigma_r_from_{e_col}'] = np.nan

    for index, row in df_merged.iterrows():
        material = row['Material']
        C_val = row['Stable_C_Coefficient']

        # 控制台按行排版，方便对比
        print(f"👉 {material: <20} | ", end="")

        for e_col in actual_e_cols:
            Er = row[e_col]

            # 容错：跳过空值或非正数
            if pd.isna(Er) or Er <= 0:
                print(f"[{e_col}: 缺失] ".ljust(15), end="")
                continue

            initial_guess = Er / 50.0

            try:
                root, infodict, ier, mesg = fsolve(
                    representative_stress_equation,
                    x0=initial_guess,
                    args=(C_val, Er),
                    full_output=True
                )

                if ier == 1:
                    sigma_r_fit = root[0]
                    df_merged.at[index, f'Sigma_r_from_{e_col}'] = sigma_r_fit
                    print(f"[{e_col}]: {sigma_r_fit:.2f} ".ljust(15), end="")
                else:
                    print(f"[{e_col}: 未收敛] ".ljust(15), end="")

            except Exception:
                print(f"[{e_col}: 异常] ".ljust(15), end="")

        print()

    # ==========================================================
    # 6. 生成全景交叉验证散点图 (sigma_y_m vs calculated sigma_r)
    # ==========================================================
    if has_yield_stress:
        print("-" * 70)
        print("📊 正在绘制屈服强度交叉验证全景拼图...")

        # 动态计算拼图的行数和列数（例如 4 种 E 会画成 2x2，3 种 E 就是 2x2 留白一张）
        n_cols = 2
        n_rows = math.ceil(len(actual_e_cols) / 2)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

        # 将子图矩阵展平以便循环遍历
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]  # 如果只有 1 个 E

        for i, e_col in enumerate(actual_e_cols):
            ax = axes[i]
            y_col = f'Sigma_r_from_{e_col}'

            # 剥离缺失值，防止画图报错
            plot_df = df_merged[['sigma_y_m', y_col]].dropna()

            if not plot_df.empty:
                # 绘制实际数据的散点
                ax.scatter(plot_df['sigma_y_m'], plot_df[y_col], color='purple', edgecolor='black', alpha=0.7, s=50,
                           label='Calculated vs Reference')

                # 获取数据范围以绘制完美的 y=x 参考线
                min_val = min(plot_df['sigma_y_m'].min(), plot_df[y_col].min()) * 0.9
                max_val = max(plot_df['sigma_y_m'].max(), plot_df[y_col].max()) * 1.1

                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6, linewidth=2,
                        label='y = x (Perfect Fit)')

                # 图表装潢
                ax.set_title(f'Using Modulus: {e_col}', fontsize=12, fontweight='bold')
                ax.set_xlabel(r'Reference Yield Strength $\sigma_{y,m}$ (GPa)', fontsize=10)
                ax.set_ylabel(r'Calculated $\sigma_r$ (GPa) ($n=0$)', fontsize=10)
                ax.legend(loc='upper left')
                ax.grid(True, linestyle=':', alpha=0.6)
            else:
                ax.set_title(f'No valid data for {e_col}')
                ax.axis('off')

        # 隐藏多余的空白子图 (如果是单数个图)
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(plot_save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✨ 全景对比图已生成: {plot_save_path}")

    # ==========================================================
    # 7. 数据列整理与安全导出
    # ==========================================================
    output_columns = ['Material', 'Stable_C_Coefficient']
    if has_yield_stress:
        output_columns.append('sigma_y_m')

    for e_col in actual_e_cols:
        output_columns.append(e_col)
        output_columns.append(f'Sigma_r_from_{e_col}')

    all_cols = [c for c in df_merged.columns if c not in output_columns and c != 'Material_Name']
    df_final = df_merged[output_columns + all_cols]

    try:
        df_final.to_csv(output_csv_path, index=False, float_format="%.4f", encoding='utf-8-sig')
        print("-" * 70)
        print(f"🎉 全部并行计算完成！")
        print(f"📁 对比结果与图像已安全存入专属档案库:\n   {run_output_dir}")
        print("=" * 70)
    except PermissionError:
        print("【错误】导出失败！CSV 文件正在被占用，请关闭 Excel 后再试。")
    except Exception as e:
        print(f"【错误】导出 CSV 失败: {e}")


if __name__ == "__main__":
    calculate_yield_stress()