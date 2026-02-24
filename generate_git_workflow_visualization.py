#!/usr/bin/env python3
"""
Git 工作流可視化生成器
自動生成當前 Git 專案的分支結構和工作流程圖
"""

import subprocess
import os
from datetime import datetime

def run_command(cmd):
    """執行 shell 命令並返回輸出"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""

def get_current_branch():
    """獲取當前分支名稱"""
    return run_command("git rev-parse --abbrev-ref HEAD")

def get_all_branches():
    """獲取所有本地分支"""
    output = run_command("git branch")
    if not output:
        return []
    branches = [b.strip().replace('* ', '') for b in output.split('\n')]
    return [b for b in branches if b]

def get_remote_branches():
    """獲取所有遠端分支"""
    output = run_command("git branch -r")
    if not output:
        return []
    branches = [b.strip() for b in output.split('\n')]
    return [b.replace('origin/', '') for b in branches if 'origin/' in b and '->' not in b]

def get_latest_commit_info():
    """獲取最新 commit 資訊"""
    commit_hash = run_command("git rev-parse --short HEAD")
    commit_msg = run_command("git log -1 --pretty=%B")
    commit_date = run_command("git log -1 --pretty=%cd --date=short")
    return commit_hash, commit_msg, commit_date

def generate_mermaid_diagram():
    """生成 Mermaid 格式的工作流圖"""
    current_branch = get_current_branch()
    local_branches = get_all_branches()
    remote_branches = get_remote_branches()
    commit_hash, commit_msg, commit_date = get_latest_commit_info()

    # 分類分支
    feature_branches = [b for b in local_branches if b.startswith('feature-')]
    test_branches = [b for b in local_branches if b.startswith('test-')]

    mermaid_code = """%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#6495ED','primaryTextColor':'#fff','primaryBorderColor':'#4169E1','lineColor':'#A9A9A9','secondaryColor':'#90EE90','tertiaryColor':'#FFD700'}}}%%
graph TB
    subgraph GitHub Remote
        ORIGIN[("🌐 origin/main<br/>穩定版本")]
    end

    subgraph Local Repository
        MAIN[("📦 main<br/>本地主分支")]
"""

    # 添加 feature 分支
    if feature_branches:
        mermaid_code += "        \n        subgraph Features[功能開發分支]\n"
        for i, branch in enumerate(feature_branches):
            branch_name = branch.replace('feature-', '')
            is_current = "🔸 " if branch == current_branch else ""
            mermaid_code += f"            F{i}[{is_current}feature-{branch_name}]\n"
        mermaid_code += "        end\n"

    # 添加 test 分支
    if test_branches:
        mermaid_code += "        \n        subgraph Tests[測試分支]\n"
        for i, branch in enumerate(test_branches):
            branch_name = branch.replace('test-', '')
            is_current = "🔸 " if branch == current_branch else ""
            mermaid_code += f"            T{i}[{is_current}test-{branch_name}]\n"
        mermaid_code += "        end\n"

    mermaid_code += "    end\n\n"

    # 添加連接關係
    mermaid_code += "    ORIGIN -.->|pull| MAIN\n"
    mermaid_code += "    MAIN -->|push| ORIGIN\n"

    # feature 分支從 main 分出
    for i in range(len(feature_branches)):
        mermaid_code += f"    MAIN -.->|branch| F{i}\n"

    # feature 分支合併到 test 分支
    for i in range(min(len(feature_branches), len(test_branches))):
        mermaid_code += f"    F{i} -.->|merge| T{i}\n"

    # test 分支合併回 main
    for i in range(len(test_branches)):
        mermaid_code += f"    T{i} -.->|測試通過| MAIN\n"

    # 添加當前狀態資訊
    mermaid_code += f"""
    INFO["📊 當前狀態<br/>分支: {current_branch}<br/>Commit: {commit_hash}<br/>日期: {commit_date}"]

    style ORIGIN fill:#6495ED,stroke:#4169E1,stroke-width:3px,color:#fff
    style MAIN fill:#32CD32,stroke:#228B22,stroke-width:2px,color:#fff
    style INFO fill:#FFE4B5,stroke:#FFA500,stroke-width:2px
"""

    return mermaid_code

def save_mermaid_to_html(mermaid_code, output_file='git-workflow-visualization.html'):
    """將 Mermaid 圖表儲存為 HTML 文件"""
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Git 工作流可視化</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: 'Arial', 'Microsoft YaHei', sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            border-bottom: 3px solid #6495ED;
            padding-bottom: 15px;
        }}
        .mermaid {{
            text-align: center;
            margin: 30px 0;
        }}
        .footer {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔄 Git 工作流程可視化</h1>
        <div class="mermaid">
{mermaid_code}
        </div>
        <div class="footer">
            自動生成於 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            專案路徑: {os.getcwd()}
        </div>
    </div>
    <script>
        mermaid.initialize({{ startOnLoad: true }});
    </script>
</body>
</html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)

    return output_file

def generate_simple_text_visualization():
    """生成簡單的文字版工作流可視化"""
    current_branch = get_current_branch()
    local_branches = get_all_branches()
    remote_branches = get_remote_branches()
    commit_hash, commit_msg, commit_date = get_latest_commit_info()

    output = []
    output.append("=" * 80)
    output.append("Git 工作流程可視化")
    output.append("=" * 80)
    output.append(f"\n📅 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"📂 專案路徑: {os.getcwd()}")
    output.append(f"🔸 當前分支: {current_branch}")
    output.append(f"📝 最新 Commit: {commit_hash} - {commit_msg[:50]}")
    output.append(f"📆 提交日期: {commit_date}")
    output.append("\n" + "-" * 80)

    output.append("\n🌐 GitHub Remote 分支:")
    if remote_branches:
        for branch in remote_branches:
            output.append(f"   └─ origin/{branch}")
    else:
        output.append("   (無遠端分支)")

    output.append("\n📦 本地分支架構:")
    output.append("   ├─ main (本地主分支)")

    # Feature 分支
    feature_branches = [b for b in local_branches if b.startswith('feature-')]
    if feature_branches:
        output.append("   │")
        output.append("   ├─ 功能開發分支:")
        for i, branch in enumerate(feature_branches):
            prefix = "   │  └─" if i == len(feature_branches) - 1 else "   │  ├─"
            marker = " 🔸 (當前)" if branch == current_branch else ""
            output.append(f"{prefix} {branch}{marker}")

    # Test 分支
    test_branches = [b for b in local_branches if b.startswith('test-')]
    if test_branches:
        output.append("   │")
        output.append("   └─ 測試分支:")
        for i, branch in enumerate(test_branches):
            prefix = "      └─" if i == len(test_branches) - 1 else "      ├─"
            marker = " 🔸 (當前)" if branch == current_branch else ""
            output.append(f"{prefix} {branch}{marker}")

    output.append("\n" + "-" * 80)
    output.append("\n📋 工作流程:")
    output.append("   1. main ← pull ← origin/main (同步遠端)")
    output.append("   2. feature-* ← branch ← main (建立功能分支)")
    output.append("   3. test-* ← merge ← feature-* (合併到測試)")
    output.append("   4. main ← merge ← test-* (測試通過後合併)")
    output.append("   5. origin/main ← push ← main (推送到遠端)")

    output.append("\n" + "=" * 80)

    return "\n".join(output)

def main():
    """主函數"""
    print("🔄 正在生成 Git 工作流可視化...")

    # 檢查是否在 Git 倉庫中
    if not os.path.exists('.git'):
        print("❌ 錯誤：當前目錄不是 Git 倉庫")
        return

    try:
        # 生成 Mermaid 圖表
        mermaid_code = generate_mermaid_diagram()
        html_file = save_mermaid_to_html(mermaid_code)
        print(f"✅ HTML 可視化已生成: {html_file}")

        # 生成文字版可視化
        text_viz = generate_simple_text_visualization()
        text_file = 'git-workflow-visualization.txt'
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_viz)
        print(f"✅ 文字版可視化已生成: {text_file}")

        # 顯示文字版
        print("\n" + text_viz)

        print("\n💡 提示：")
        print(f"   - 在瀏覽器中開啟 {html_file} 查看互動式圖表")
        print(f"   - 使用 cat {text_file} 查看文字版")

    except Exception as e:
        print(f"❌ 生成失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
