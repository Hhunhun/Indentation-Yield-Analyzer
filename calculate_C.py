import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
from datetime import datetime
import math

# 忽略常规解析警告，保持控制台干净
warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)


def ideal_berkovich(Pd, C):
    """
    理想 Berkovich 压头加载段的 Kick 定律物理模型
    F = C * h^2
    这里强制锁定 m=2，使得提取出的 C 具有纯粹的应力量纲。
    """
    return C * (Pd ** 2)


def process_single_material(file_path, run_output_dir, global_summary_path, plot_individual=True):
    """
    处理单一材料的数据：包括读取、拆分、清洗、拟合、稳态识别及绘图
    """
    # 提取材料名称 (不带后缀)
    material_name = os.path.splitext(os.path.basename(file_path))[0]

    # 构建当前材料专属的输出目录 (直接存放在本次运行的 Timestamp 文件夹下)
    main_output_dir = os.path.join(run_output_dir, material_name)
    os.makedirs(main_output_dir, exist_ok=True)

    if plot_individual:
        # 如果需要保存单组图，在材料文件夹下再建一个子文件夹
        plots_sub_dir = os.path.join(main_output_dir, "Individual_Fits")
        os.makedirs(plots_sub_dir, exist_ok=True)

    summary_txt_path = os.path.join(main_output_dir, f"{material_name}_Results.txt")

    # 1. 稳健读取机制 (多编码轮询，防止仪器导出乱码)
    encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'ansi', 'latin1']
    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(file_path, sep=r'\s+', header=None, comment='#', engine='python', encoding=enc)
            break
        except Exception:
            continue

    if df is None:
        print(f"  ❌ 【失败】无法解析: {material_name}")
        return None

    # 2. 智能分组 (识别多列并排或垂直堆叠)
    groups = []
    if df.shape[1] >= 4:
        # 多列并排情况 (如 Pd1, Fn1, Pd2, Fn2...)
        for i in range(0, df.shape[1] - 1, 2):
            sub_df = df.iloc[:, i:i + 2].apply(pd.to_numeric, errors='coerce').dropna()
            if len(sub_df) > 10:
                groups.append((sub_df.iloc[:, 0].values, sub_df.iloc[:, 1].values))
    elif df.shape[1] in [2, 3]:
        # 垂直堆叠情况，利用空行或压深断点切割
        df_num = df.iloc[:, 0:2].apply(pd.to_numeric, errors='coerce')
        is_nan_row = df_num.isna().any(axis=1)
        group_ids = is_nan_row.cumsum()
        for _, group_df in df_num[~is_nan_row].groupby(group_ids):
            if len(group_df) < 10: continue
            Pd_arr = group_df.iloc[:, 0].values
            Fn_arr = group_df.iloc[:, 1].values
            # 差分法寻找位移突然归零的断点
            drops = np.where(np.diff(Pd_arr) < -0.2 * np.max(Pd_arr))[0]
            if len(drops) > 0:
                split_indices = [0] + list(drops + 1) + [len(Pd_arr)]
                for j in range(len(split_indices) - 1):
                    start, end = split_indices[j], split_indices[j + 1]
                    if end - start > 10:
                        groups.append((Pd_arr[start:end], Fn_arr[start:end]))
            else:
                groups.append((Pd_arr, Fn_arr))

    # 3. 核心拟合运算
    results = []
    for idx, (Pd_raw, Fn_raw) in enumerate(groups, start=1):
        # 截取加载段 (最大载荷之前的点)
        max_idx = np.argmax(Fn_raw)
        Pd_load = Pd_raw[:max_idx + 1]
        Fn_load = Fn_raw[:max_idx + 1]

        valid_mask = (Pd_load > 0) & (Fn_load > 0)
        Pd_valid = Pd_load[valid_mask]
        Fn_valid = Fn_load[valid_mask]

        if len(Pd_valid) < 5: continue

        Fmax = np.max(Fn_valid)
        initial_guess = [Fmax / (np.max(Pd_valid) ** 2)]

        try:
            # 执行非线性最小二乘法拟合
            popt, _ = curve_fit(ideal_berkovich, Pd_valid, Fn_valid, p0=initial_guess, bounds=([0], [np.inf]))
            C_fit = popt[0]
            Fn_pred = ideal_berkovich(Pd_valid, C_fit)
            r_squared = r2_score(Fn_valid, Fn_pred)
            results.append((idx, Fmax, C_fit, 2.0, r_squared))

            # 单组拟合图静默绘制与保存
            if plot_individual:
                plt.figure(figsize=(5, 4))
                plt.scatter(Pd_valid, Fn_valid, color='blue', s=5, alpha=0.5)
                plt.plot(Pd_valid, Fn_pred, color='red', linewidth=1.5)
                plt.title(f'{material_name} - Group {idx:02d}')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.savefig(os.path.join(plots_sub_dir, f"Fit_Group_{idx:02d}.png"), dpi=100, bbox_inches='tight')
                plt.close('all')
        except Exception:
            continue

    if not results:
        return None

    # 按最大载荷 Fmax 从小到大排序
    sorted_results = sorted(results, key=lambda x: x[1])
    Fmax_vals = np.array([r[1] for r in sorted_results])
    C_vals = np.array([r[2] for r in sorted_results])

    # ==========================================================
    # ★ 异常点清洗 (Outlier Purge) - 防护离谱起飞数据
    # ==========================================================
    median_C = np.median(C_vals)
    # 物理防线：若 C 值大得超过中位数的 5 倍，必定是试验发散点
    valid_mask_outliers = C_vals < (5 * median_C)
    has_outliers = not np.all(valid_mask_outliers)

    # 存在死点时，输出一张清洗前的数据保留证据
    if has_outliers and plot_individual:
        plt.figure(figsize=(7, 5))
        plt.scatter(Fmax_vals, C_vals, color='red', edgecolor='black', alpha=0.7, label='Raw Data (with outliers)')
        plt.xlabel('Maximum Load $F_{max}$')
        plt.ylabel('Fitted Coefficient $C$')
        plt.title(f'{material_name}: Before Outlier Removal')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(main_output_dir, f"{material_name}_1_Before_Outlier_Removal.png"), dpi=150,
                    bbox_inches='tight')
        plt.close('all')

    # 应用清洗掩码
    Fmax_vals = Fmax_vals[valid_mask_outliers]
    C_vals = C_vals[valid_mask_outliers]

    # 导出单材料总数据表 (剔除极端异常值)
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write("Group_ID\tFmax\tC_Coefficient\tm_Exponent\tR_Squared\n")
        for res in sorted_results:
            if res[2] < (5 * median_C):
                f.write(f"{res[0]}\t{res[1]:.6e}\t{res[2]:.6e}\t{res[3]:.1f}\t{res[4]:.6f}\n")

    # ==========================================================
    # ★ 终极平台算法：中段锚定与双向截断 (排除 <100 和大载荷掉尾)
    # ==========================================================
    if len(C_vals) < 3:
        return None

    # 1. 强制硬截断：只考虑 Fmax >= 100 的点作为有效候选
    candidate_mask = Fmax_vals >= 100

    # 2. 寻找绝对可靠的中段基准 (100 <= Fmax <= 300 区间的最稳点)
    mid_mask = (Fmax_vals >= 100) & (Fmax_vals <= 300)

    if np.sum(mid_mask) >= 3:
        baseline_C = np.median(C_vals[mid_mask])
    elif np.sum(candidate_mask) >= 1:
        baseline_C = np.median(C_vals[candidate_mask])
    else:
        baseline_C = np.median(C_vals)  # 保底防御

    # 3. 动态掩码过滤：Fmax >= 100 且 C 值偏离基准不超过 8% (精准切除掉尾)
    stable_mask = candidate_mask & (np.abs(C_vals - baseline_C) / max(baseline_C, 1e-6) <= 0.08)

    # 如果要求太严导致可用点太少，放宽容差至 15%
    if np.sum(stable_mask) < 3:
        stable_mask = candidate_mask & (np.abs(C_vals - baseline_C) / max(baseline_C, 1e-6) <= 0.15)

    # 终极防御：如果全军覆没，则全盘采纳
    if np.sum(stable_mask) == 0:
        stable_mask = np.ones_like(Fmax_vals, dtype=bool)

        # 计算精准稳态均值
    C_stable = np.mean(C_vals[stable_mask])
    Fmax_stable_min = np.min(Fmax_vals[stable_mask])
    Fmax_stable_max = np.max(Fmax_vals[stable_mask])

    # 追加记录到全局总览表中 (保留两位小数)
    with open(global_summary_path, 'a', encoding='utf-8') as gf:
        gf.write(f"{material_name}\t{C_stable:.2f}\n")

    # ==========================================================
    # 绘制单材料清洗/截断处理后最终散点图
    # ==========================================================
    if plot_individual:
        plt.figure(figsize=(7, 5))

        # 灰色：被抛弃的点 (包括左侧浅层 <100，以及右侧掉尾巴的点)
        plt.scatter(Fmax_vals[~stable_mask], C_vals[~stable_mask],
                    color='gray', edgecolor='black', alpha=0.5, label='Discarded (ISE or Tail Drop)')
        # 紫色：核心稳定平台区
        plt.scatter(Fmax_vals[stable_mask], C_vals[stable_mask],
                    color='purple', edgecolor='black', alpha=0.8, label='Stable Region')

        # 灰色虚线标定稳定区
        plt.hlines(y=C_stable, xmin=Fmax_stable_min, xmax=Fmax_stable_max,
                   colors='gray', linestyles='dashed', linewidth=2,
                   label=f'Stable $C \\approx {C_stable:.2f}$')

        plt.xlabel('Maximum Load $F_{max}$')
        plt.ylabel('Fitted Coefficient $C$')
        plt.title(f'{material_name}: $F_{{max}}$ vs $C$ ($m=2$)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        save_name = f"{material_name}_2_After_Outlier_Removal.png" if has_outliers else f"{material_name}_Fmax_vs_C.png"
        plt.savefig(os.path.join(main_output_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close('all')

    return {
        'name': material_name,
        'Fmax': Fmax_vals,
        'C': C_vals,
        'stable_mask': stable_mask,
        'C_stable': C_stable,
        'has_outliers': has_outliers
    }


# ---------------------------------------------------------
# 交互式调度中枢 (更名为 calculate_C)
# ---------------------------------------------------------
def calculate_C():
    # ================= 关键配置区 =================
    # 原始 30 种材料的存放路径
    data_dir = r"D:\Woooooork\berkovich_C\data\Pd_Fn"
    # 输出根目录
    base_output_dir = r"D:\Woooooork\berkovich_C\output\yao_C"
    # ==============================================

    # 提取当前时间戳，为每次运行创建独立的总档案库
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(base_output_dir, f"Run_{time_str}")
    os.makedirs(run_output_dir, exist_ok=True)

    # 汇总文件现在直接放置于本次运行的专属目录下
    global_summary_path = os.path.join(run_output_dir, "Global_Stable_C_Summary.txt")

    search_pattern = os.path.join(data_dir, "*.[tT][xX][tT]")
    txt_files = glob.glob(search_pattern)
    total_files = len(txt_files)

    if total_files == 0:
        print(f"【警告】在 {data_dir} 目录下没有找到任何 txt 文件！请检查路径。")
        return

    print("=" * 55)
    print(f"🔬 终极非晶纳米压入分析引擎 (检测到 {total_files} 个材料)")
    print("=" * 55)
    for i, f in enumerate(txt_files, start=1):
        print(f" [{i:02d}] {os.path.basename(f)}")
    print("-" * 55)
    print("👉 请选择运行模式：")
    print(" - 输入【数字序号】(如 1) 单独跑某一个材料作算法调试")
    print(" - 输入【ALL】一键执行目录下所有材料的批量处理")
    print(" - 输入【Q】退出程序")

    choice = input("您的选择: ").strip().upper()

    if choice == 'Q':
        return
    elif choice == 'ALL':
        print("\n🤔 检测到批量运行命令。")
        only_summary = input(
            "是否仅输出一张包含所有材料的 Fmax-C 汇总拼图，以加快运行速度？(Y/N): ").strip().upper() == 'Y'

        with open(global_summary_path, 'w', encoding='utf-8') as gf:
            gf.write("Material_Name\tStable_C_Coefficient\n")

        success_count = 0
        all_materials_data = []

        for i, file_path in enumerate(txt_files, start=1):
            name = os.path.basename(file_path)
            print(f"正在处理 ({i:02d}/{total_files}): {name} ...", end="")

            plot_indiv = not only_summary
            # 传入本次运行的专属目录 run_output_dir
            res_data = process_single_material(file_path, run_output_dir, global_summary_path,
                                               plot_individual=plot_indiv)

            if res_data is not None:
                all_materials_data.append(res_data)
                flag = " (已清洗异常点)" if res_data['has_outliers'] else ""
                print(f" ✅ 完成{flag}")
                success_count += 1
            else:
                print(" ⚠️ 失败")

        # ==============================================
        # 绘制终极全局汇总拼图 (Global Grid Plot)
        # ==============================================
        if all_materials_data:
            print("\n正在绘制全材料汇总拼图，请稍候...")
            num_mats = len(all_materials_data)
            cols = 5
            rows = math.ceil(num_mats / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
            axes = axes.flatten()

            for i, data in enumerate(all_materials_data):
                ax = axes[i]
                F = data['Fmax']
                C = data['C']
                stable_mask = data['stable_mask']
                C_st = data['C_stable']

                # 画散点 (被抛弃的标灰，稳定的标紫)
                ax.scatter(F[~stable_mask], C[~stable_mask], color='gray', alpha=0.5, s=12)
                ax.scatter(F[stable_mask], C[stable_mask], color='purple', alpha=0.8, s=12)

                # 画基准线 (灰色虚线，保留两位小数)
                if np.any(stable_mask):
                    ax.hlines(y=C_st, xmin=np.min(F[stable_mask]), xmax=np.max(F[stable_mask]),
                              colors='gray', linestyles='dashed', linewidth=1.5,
                              label=f'C ≈ {C_st:.2f}')

                ax.set_title(data['name'], fontsize=9)
                ax.legend(fontsize=8)
                ax.grid(True, linestyle=':', alpha=0.6)

            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            # 汇总拼图直接保存在本次运行目录下
            grid_save_path = os.path.join(run_output_dir, f"All_Materials_Grid_Summary.png")
            plt.savefig(grid_save_path, dpi=200, bbox_inches='tight')
            plt.close('all')
            print(f"✨ 汇总拼图已生成: {grid_save_path}")

        print("\n" + "=" * 55)
        print("🎉 批量运算全部结束！")
        print(f"📊 成功分析: {success_count}/{total_files} 个材料。")
        print(f"📁 本次运行结果已全部储存于:\n   {run_output_dir}")
        print("=" * 55)

    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < total_files:
                target_file = txt_files[idx]
                name = os.path.basename(target_file)
                print(f"\n⚙️ 正在单独调试: {name} ...")

                if not os.path.exists(global_summary_path):
                    with open(global_summary_path, 'w', encoding='utf-8') as gf:
                        gf.write("Material_Name\tStable_C_Coefficient\n")

                res_data = process_single_material(target_file, run_output_dir, global_summary_path,
                                                   plot_individual=True)
                if res_data is not None:
                    print("✅ 单点调试完成！")
                    print(f"📁 结果路径: {run_output_dir}")
                    if res_data['has_outliers']:
                        print("💡 提示：该材料存在异常点，程序已为您保存了剔除前后的对比图。")
                else:
                    print("⚠️ 处理失败。")
            else:
                print("【错误】输入的序号超出文件列表范围。")
        except ValueError:
            print("【错误】输入无效，请输入数字或 ALL。")


if __name__ == "__main__":
    calculate_C()