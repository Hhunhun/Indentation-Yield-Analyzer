import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import warnings
from datetime import datetime  # 用于生成时间戳

# 忽略数值计算中可能产生的除零警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================================
# ★ 关键参数配置区
# ==========================================================
# 默认填入的是论文中 60° 压头的值。
# 【强烈建议】：请务必查阅您的 image_4adf00.png 截图，
# 将下面四个常数修改为 Berkovich 压头（70.3°）对应的值！
C1 = 0.0646  # 对应 ln 的 3 次方系数
C2 = -2.210  # 对应 ln 的 2 次方系数
C3 = 21.589  # 对应 ln 的 1 次方系数
C4 = -28.571  # 常数项


def representative_stress_equation(sigma_r, C, Er):
    """
    根据姚博论文 CR-EMI 方法公式 (3) 构造的隐式方程：
    C / sigma_r = c1*(ln(Er/sigma_r))^3 + c2*(ln(Er/sigma_r))^2 + c3*ln(Er/sigma_r) + c4
    """
    # 物理防线：应力不能为负或零
    if sigma_r <= 0:
        return 1e9

    # 核心计算：提取对数项
    ln_term = np.log(Er / sigma_r)
    # 计算公式右侧多项式
    right_side = C1 * (ln_term ** 3) + C2 * (ln_term ** 2) + C3 * ln_term + C4

    # 返回等式两边的差值，供 fsolve 寻找零点（当返回值为 0 时，sigma_r 即为所求）
    return right_side - (C / sigma_r)


def calculate_sigma_r():
    print("=" * 55)
    print("🚀 启动 CR-EMI 表征应力联合求解引擎")
    print("=" * 55)

    # ==========================================================
    # 1. 目录架构配置与时间戳生成
    # ==========================================================
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 基础目录配置
    data_dir = r"D:\Woooooork\berkovich_C\data"
    base_output_dir = r"D:\Woooooork\berkovich_C\output\yao_sigma_r"
    yao_c_output_dir = r"D:\Woooooork\berkovich_C\output\yao_C"

    # 每次运行生成独立的带时间戳的子文件夹，保持工作区高度整洁
    run_output_dir = os.path.join(base_output_dir, f"Run_{time_str}")
    os.makedirs(run_output_dir, exist_ok=True)

    # 已知模量数据表路径
    known_data_path = os.path.join(data_dir, "knowndata.csv")
    # 最终结果输出路径 (存放于本次运行的专属子文件夹中)
    output_csv_path = os.path.join(run_output_dir, "Calculated_Sigma_r_Results.csv")

    # ==========================================================
    # 2. 智能检索最新生成的 C 值汇总表
    # ==========================================================
    # 自动去 yao_C 文件夹下寻找最新的 Run_xxx 文件夹
    run_folders = glob.glob(os.path.join(yao_c_output_dir, "Run_*"))
    if not run_folders:
        print(f"【错误】在 {yao_c_output_dir} 下未找到任何 C 值计算记录，请先运行 calculate_C 程序。")
        return

    # 按照文件夹创建时间排序，获取最新的一次运行记录
    latest_run_dir = max(run_folders, key=os.path.getctime)
    c_summary_path = os.path.join(latest_run_dir, "Global_Stable_C_Summary.txt")

    if not os.path.exists(c_summary_path):
        print(f"【错误】在最新的记录 {latest_run_dir} 中未找到 Global_Stable_C_Summary.txt文件。")
        return

    print(f"🔍 智能检索：已自动锁定最新的 C 值汇总表 -> {os.path.basename(latest_run_dir)}")

    # ==========================================================
    # 3. 稳健读取数据
    # ==========================================================
    # 极致稳健的 CSV 读取机制（解决 Excel 另存为 CSV 带来的编码问题）
    encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'ansi', 'latin1']
    df_known = None
    for enc in encodings_to_try:
        try:
            df_known = pd.read_csv(known_data_path, encoding=enc)
            print(f"成功使用 '{enc}' 编码破解并读取 known_data 表格！")
            break
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            print(f"【错误】找不到文件，请确认 {known_data_path} 是否存在。")
            return

    if df_known is None:
        print("【致命错误】所有常见编码均无法解析 knowndata.csv 文件。")
        return

    try:
        # 清理列名和材料名中可能包含的隐藏空格，确保 merge 时精准匹配
        df_known.columns = df_known.columns.str.strip()
        df_known['Material'] = df_known['Material'].astype(str).str.strip()

        # 读取最新的 TXT 数据 (之前生成的稳定 C 值结果表)
        df_c = pd.read_csv(c_summary_path, sep='\t')
        df_c.columns = df_c.columns.str.strip()
        df_c['Material_Name'] = df_c['Material_Name'].astype(str).str.strip()

    except Exception as e:
        print(f"【错误】读取提取的 C 值汇总表失败: {e}")
        return

    # ==========================================================
    # 4. 自动化数据融合 (Merge)
    # ==========================================================
    df_merged = pd.merge(df_known, df_c, left_on='Material', right_on='Material_Name', how='inner')

    if df_merged.empty:
        print("【错误】数据合并失败！请检查两个表中的材料名称是否完全对应一致。")
        return

    print(f"✅ 成功匹配并融合了 {len(df_merged)} 种材料的数据。开始执行数值求解...")

    # ==========================================================
    # 5. 遍历求解表征应力
    # ==========================================================
    sigma_r_results = []

    for index, row in df_merged.iterrows():
        material = row['Material']
        # 提取 Er 和 C
        Er = row['Er']
        C_val = row['Stable_C_Coefficient']

        # 初始猜测值：根据金属材料特性，表征应力通常是弹性模量的 1/50 左右
        initial_guess = Er / 50.0

        try:
            # 使用 scipy.optimize.fsolve 进行数值求根
            root, infodict, ier, mesg = fsolve(
                representative_stress_equation,
                x0=initial_guess,
                args=(C_val, Er),
                full_output=True
            )

            if ier == 1:
                sigma_r_fit = root[0]
                sigma_r_results.append(sigma_r_fit)
                print(f"  -> {material: <25}: 求解成功 | Er={Er:.2f}, C={C_val:.2f} => σ_r = {sigma_r_fit:.4f}")
            else:
                sigma_r_results.append(np.nan)
                print(f"  ⚠️ {material: <25}: 算法未收敛 ({mesg})")

        except Exception as e:
            sigma_r_results.append(np.nan)
            print(f"  ⚠️ {material: <25}: 求解发生异常 ({e})")

    # ==========================================================
    # 6. 数据整理与安全导出
    # ==========================================================
    df_merged['Sigma_r_Calculated'] = sigma_r_results

    # 整理输出列，将关键的计算结果放在最前面，剔除冗余重名的列
    output_columns = ['Material', 'Er', 'Stable_C_Coefficient', 'Sigma_r_Calculated']
    all_cols = [c for c in df_merged.columns if c not in output_columns and c != 'Material_Name']
    df_final = df_merged[output_columns + all_cols]

    try:
        # 使用 utf-8-sig 编码导出，保证用 Excel 软件双击直接打开时中文等字符不会乱码
        df_final.to_csv(output_csv_path, index=False, float_format="%.4f", encoding='utf-8-sig')
        print("=" * 55)
        print(f"🎉 全部计算完成！")
        print(f"📁 本次运行结果已安全存入专属档案库:\n   {run_output_dir}")
        print("=" * 55)
    except PermissionError:
        print("【错误】导出失败！请检查该 CSV 文件是否正在被其他程序（如 Excel）打开占用，关闭后再试。")
    except Exception as e:
        print(f"【错误】导出 CSV 失败: {e}")


if __name__ == "__main__":
    calculate_sigma_r()