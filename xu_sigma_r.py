import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# 忽略计算中可能产生的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================================
# ★ 关键参数配置区 (Xu 模型)
# ==========================================================
# 公式: ln(he/hmax) = 0.7189 * ln(sigma_y / E) + 1.2015
# 其中 he_hmax = 1 - hp_hmax
COEF_A = 0.7189
COEF_B = 1.2015

# 根据要求，仅处理 Er 和 Er_O&P 两种模量
E_COLUMNS_TO_PROCESS = ['Er', 'Er_O&P']


def calculate_xu_yield_stress(hp_hmax, E):
    """
    根据 Xu 的公式，直接推导出的解析解：
    sigma_y = E * exp( (ln(he/hmax) - 1.2015) / 0.7189 )
    """
    he_hmax = 1.0 - hp_hmax

    # 物理防线：压深比必须在 0 到 1 之间，否则对数函数无意义
    if he_hmax <= 0 or he_hmax >= 1:
        raise ValueError("Invalid hp/hmax value (must be between 0 and 1)")

    # 显式计算，绝对稳定，无需 fsolve
    ln_term = np.log(he_hmax)
    power_term = (ln_term - COEF_B) / COEF_A
    sigma_y = E * np.exp(power_term)

    return sigma_y


def run_xu_model():
    print("=" * 70)
    print("🚀 启动 Xu 模型 - 屈服强度多模量显式求解引擎")
    print("=" * 70)

    # ==========================================================
    # 1. 目录架构配置与时间戳生成
    # ==========================================================
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_dir = r"D:\Woooooork\berkovich_C\data"
    # 输出到专门的 xu_sigma_r 文件夹下
    base_output_dir = r"D:\Woooooork\berkovich_C\output\xu_sigma_r"

    run_output_dir = os.path.join(base_output_dir, f"Run_{time_str}")
    os.makedirs(run_output_dir, exist_ok=True)

    known_data_path = os.path.join(data_dir, "knowndata.csv")
    output_csv_path = os.path.join(run_output_dir, "Calculated_Sigma_y_Xu_Results.csv")
    plot_save_path = os.path.join(run_output_dir, "Yield_Strength_Comparison_Xu.png")

    # ==========================================================
    # 2. 稳健读取数据 (Xu 模型不需要 C 值汇总表)
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
        print("【错误】在 CSV 中找不到关键列 'hp_hmax'，Xu 模型无法计算！")
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
    # 3. 遍历求解屈服强度 (解析解直接计算)
    # ==========================================================
    for e_col in actual_e_cols:
        df_known[f'Sigma_y_from_{e_col}'] = np.nan

    for index, row in df_known.iterrows():
        material = row['Material']
        hp_hmax = row['hp_hmax']

        # 如果 hp_hmax 数据缺失或物理上不合理，直接跳过
        if pd.isna(hp_hmax) or hp_hmax <= 0 or hp_hmax >= 1:
            print(f"👉 {material: <20} | [hp_hmax 异常跳过]")
            continue

        print(f"👉 {material: <20} | hp/hmax={hp_hmax:.3f} | ", end="")

        for e_col in actual_e_cols:
            E_val = row[e_col]

            # 容错：跳过模量空值或非正数
            if pd.isna(E_val) or E_val <= 0:
                print(f"[{e_col}: 缺失] ".ljust(15), end="")
                continue

            try:
                # 调用显式解析解
                sigma_y_calc = calculate_xu_yield_stress(hp_hmax, E_val)
                df_known.at[index, f'Sigma_y_from_{e_col}'] = sigma_y_calc
                print(f"[{e_col}]: {sigma_y_calc:.2f} ".ljust(15), end="")
            except Exception as e:
                print(f"[{e_col}: 异常] ".ljust(15), end="")

        print()

    # ==========================================================
    # 4. 生成交叉验证散点图 (sigma_y_m vs calculated sigma_y)
    # ==========================================================
    if has_yield_stress:
        print("-" * 70)
        print("📊 正在绘制 Xu 模型屈服强度交叉验证拼图...")

        # 因为只配置了2种 E，绘制 1行 x 2列 的图表
        n_cols = 2
        n_rows = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5))

        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for i, e_col in enumerate(actual_e_cols):
            ax = axes[i]
            y_col = f'Sigma_y_from_{e_col}'

            # 剥离缺失值
            plot_df = df_known[['sigma_y_m', y_col]].dropna()

            if not plot_df.empty:
                # 散点绘制 (Xu 模型使用 DodgerBlue 色调区分)
                ax.scatter(plot_df['sigma_y_m'], plot_df[y_col], color='dodgerblue', edgecolor='black', alpha=0.7, s=50,
                           label='Xu Model vs Reference')

                # 动态获取 y=x 虚线范围
                min_val = min(plot_df['sigma_y_m'].min(), plot_df[y_col].min()) * 0.9
                max_val = max(plot_df['sigma_y_m'].max(), plot_df[y_col].max()) * 1.1

                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6, linewidth=2,
                        label='y = x (Perfect Fit)')

                # 图表装潢
                ax.set_title(f'Xu Model - Using Modulus: {e_col}', fontsize=11, fontweight='bold')
                ax.set_xlabel(r'Reference Yield Strength $\sigma_{y,m}$ (GPa)', fontsize=10)
                ax.set_ylabel(r'Calculated $\sigma_y$ (GPa)', fontsize=10)
                ax.legend(loc='upper left')
                ax.grid(True, linestyle=':', alpha=0.6)
            else:
                ax.set_title(f'No valid data for {e_col}')
                ax.axis('off')

        # 隐藏多余子图防错
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
        output_columns.append(f'Sigma_y_from_{e_col}')

    all_cols = [c for c in df_known.columns if c not in output_columns]
    df_final = df_known[output_columns + all_cols]

    try:
        df_final.to_csv(output_csv_path, index=False, float_format="%.4f", encoding='utf-8-sig')
        print("-" * 70)
        print(f"🎉 Xu 模型并行计算全部完成！")
        print(f"📁 结果与图像已安全存入专属档案库:\n   {run_output_dir}")
        print("=" * 70)
    except PermissionError:
        print("【错误】导出失败！CSV 文件正在被占用，请关闭 Excel 后再试。")
    except Exception as e:
        print(f"【错误】导出 CSV 失败: {e}")


if __name__ == "__main__":
    run_xu_model()