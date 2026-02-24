# Git 工作流程說明

## 概述
本專案採用客製化的 Git 工作流，確保程式碼品質與功能穩定性。

## 分支架構

```
GitHub Remote (origin)
└── main (master) ← 已測試並確認的穩定版本

Local Repository
├── main (master) ← 同步 GitHub，作為本地開發基礎
├── feature-* ← 功能開發分支
└── test-* ← 功能測試分支
```

## 分支說明

### 1. GitHub Remote - main (origin/main)
- **用途**：存放已確認且經過測試的穩定版本
- **保護規則**：只接受經過測試的功能合併
- **更新方式**：從 `test-*` 分支測試通過後合併

### 2. Local - main
- **用途**：同步 GitHub 最新版本，供本地開發者建立功能分支
- **操作**：定期從 `origin/main` pull 最新版本
- **不直接開發**：始終保持與遠端同步

### 3. feature-* 分支
- **命名規則**：`feature-功能名稱`
- **用途**：開發新功能
- **現有功能範例**：
  - `feature-docker-container` - Docker 容器功能
  - `feature-人體追蹤硬體` - 人體追蹤外部硬體
  - `feature-姿態辨識-robot控制` - 姿態辨識控制機器人（規劃中）

### 4. test-* 分支
- **命名規則**：`test-功能名稱`
- **用途**：測試對應的 feature 分支功能
- **流程**：feature 開發完成 → 合併到 test 分支 → 進行測試

## 工作流程

### 開發新功能

```bash
# 1. 確保本地 main 是最新版本
git checkout main
git pull origin main

# 2. 從 main 建立功能分支
git checkout -b feature-新功能名稱

# 3. 開發功能（進行多次 commit）
git add .
git commit -m "feat: 實作新功能 XXX"

# 4. 功能開發完成後，建立測試分支
git checkout -b test-新功能名稱

# 5. 在測試分支進行測試
# ... 執行測試、修復問題 ...
git commit -m "test: 完成新功能測試"
```

### 合併測試通過的功能到 GitHub

```bash
# 1. 確保測試分支所有測試都通過
git checkout test-新功能名稱

# 2. 切換到本地 main 並合併測試分支
git checkout main
git merge test-新功能名稱 --no-ff -m "merge: 合併新功能到 main"

# 3. 推送到 GitHub
git push origin main

# 4. （可選）刪除已合併的分支
git branch -d feature-新功能名稱
git branch -d test-新功能名稱
```

## Commit 訊息規範

使用語義化提交訊息（Semantic Commit Messages）：

- `feat:` - 新功能
- `fix:` - 錯誤修復
- `test:` - 測試相關
- `docs:` - 文檔更新
- `refactor:` - 重構程式碼
- `style:` - 程式碼格式調整
- `chore:` - 其他雜項

範例：
```
feat: 新增姿態辨識控制機器人功能
fix: 修復人體追蹤硬體連線問題
test: 完成 Docker 容器功能測試
docs: 更新 README 使用說明
```

## 工作流可視化

每次 commit 後，會自動生成工作流可視化圖片：`git-workflow-visualization.png`

此圖片**不會**上傳到 GitHub（已在 .gitignore 中排除），僅供本地參考。

## 常用指令速查

```bash
# 查看所有分支
git branch -a

# 查看當前分支狀態
git status

# 查看 commit 歷史圖形化
git log --oneline --graph --all

# 同步遠端最新版本
git pull origin main

# 切換分支
git checkout <分支名稱>

# 建立並切換到新分支
git checkout -b <新分支名稱>

# 查看工作流可視化
xdg-open git-workflow-visualization.png
```

## 注意事項

1. **永遠不要直接在 main 分支開發**
2. **功能必須經過 test 分支測試才能合併到 GitHub**
3. **定期從 origin/main pull 最新版本**
4. **使用語義化 commit 訊息**
5. **保持 commit 原子性（一個 commit 做一件事）**

## 分支保護建議

在 GitHub 上設定分支保護規則：

1. 前往 GitHub Repository → Settings → Branches
2. 新增分支保護規則（Branch protection rule）
3. 設定 `main` 分支：
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass before merging
   - ✅ Include administrators

---

最後更新：2026-02-24
