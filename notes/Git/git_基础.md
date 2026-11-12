# 🐙 Git 常用指令速查手册

> 面向日常开发的高频 Git 指令合集，包含 **基础说明 + 用法 + 实战案例**。适合收藏当速查表使用。

---

## 📑 目录

1. [仓库初始化与克隆](#1-仓库初始化与克隆)
2. [查看状态与差异](#2-查看状态与差异)
3. [暂存与提交](#3-暂存与提交)
4. [远程仓库操作](#4-远程仓库操作)
5. [分支管理](#5-分支管理)
6. [撤销与回退](#6-撤销与回退)
7. [日志与历史](#7-日志与历史)
8. [标签管理](#8-标签管理)
9. [暂存工作区（stash）](#9-暂存工作区stash)
10. [冲突处理](#10-冲突处理)
11. [常见工作流示例](#11-常见工作流示例)

---

## 1. 仓库初始化与克隆

### `git init`
**基础**：在当前目录初始化一个新的 Git 仓库。  
**用法**：`git init [目录名]`  
**案例**：
```bash
# 在当前目录初始化
git init

# 新建一个项目目录并初始化
git init my_project
```

### `git clone`
**基础**：从远程仓库克隆一份完整副本到本地。  
**用法**：`git clone <仓库地址> [本地目录名]`  
**案例**：
```bash
# 克隆仓库到同名目录
git clone https://github.com/user/RS_Unet.git

# 克隆并指定本地目录名
git clone https://github.com/user/RS_Unet.git rs_unet_local
```

---

## 2. 查看状态与差异

### `git status`
**基础**：查看工作区和暂存区当前状态（哪些文件被修改 / 已暂存 / 未跟踪）。  
**用法**：`git status [-s]`  
**案例**：
```bash
# 完整状态信息
git status

# 简洁模式（推荐日常用）
git status -s
# 输出示例：
#  M rsunet/engine.py      ← 已修改未暂存
# A  notes/day6_engine/    ← 已暂存
# ?? new_file.txt          ← 未跟踪
```

### `git diff`
**基础**：查看文件改动内容。默认对比 **工作区 vs 暂存区**。  
**用法**：`git diff [选项] [文件]`  
**案例**：
```bash
# 查看未暂存的改动
git diff

# 查看已暂存（staged）的改动
git diff --staged

# 只看某个文件的改动
git diff rsunet/engine.py

# 对比两个提交
git diff abc123 def456
```

---

## 3. 暂存与提交

### `git add`
**基础**：将工作区改动加入暂存区（staging area）。  
**用法**：`git add <文件/目录>`  
**案例**：
```bash
# 指定文件（推荐，精准可控）
git add rsunet/engine.py notes/day6_engine/

# 添加当前目录所有改动
git add .

# 交互式选择要暂存的部分（颗粒度最细）
git add -p
```

> 💡 **建议**：尽量指定文件 add，避免 `git add .` 把无关文件（日志、权重、数据）也提交进去。

### `git commit`
**基础**：把暂存区内容提交到本地仓库，形成一次版本记录。  
**用法**：`git commit -m "提交说明"`  
**案例**：
```bash
# 常规提交
git commit -m "添加单轮训练引擎与 Day6 笔记"

# 跳过暂存区，直接提交已跟踪文件的改动
git commit -am "修复 metrics 计算 bug"

# 修改上一次提交的说明（未推送前使用）
git commit --amend -m "更准确的提交说明"
```

> ✍️ **提交信息建议**：说明 **做了什么 + 为什么**，便于日后回溯。

---

## 4. 远程仓库操作

### `git remote`
**基础**：管理远程仓库地址。  
**用法**：`git remote [-v] [add/remove]`  
**案例**：
```bash
# 查看已配置的远程仓库
git remote -v

# 添加远程仓库
git remote add origin https://github.com/user/RS_Unet.git

# 修改远程地址
git remote set-url origin https://github.com/user/new_repo.git
```

### `git push`
**基础**：把本地提交推送到远程仓库。  
**用法**：`git push <远程名> <分支名>`  
**案例**：
```bash
# 推送 main 分支到 origin
git push origin main

# 首次推送并建立跟踪关系
git push -u origin main

# 强制推送（⚠️ 谨慎，会覆盖远程历史）
git push origin main --force
```

### `git pull`
**基础**：拉取远程更新并合并到当前分支（= fetch + merge）。  
**用法**：`git pull <远程名> <分支名>`  
**案例**：
```bash
# 拉取并合并 origin/main
git pull origin main

# 拉取时用 rebase 方式，保持线性历史
git pull --rebase origin main
```

### `git fetch`
**基础**：只下载远程更新，**不自动合并**，更安全。  
**用法**：`git fetch [远程名]`  
**案例**：
```bash
# 拉取所有远程更新到本地引用
git fetch origin

# 查看远程分支和本地分支的差异
git log HEAD..origin/main --oneline
```

---

## 5. 分支管理

### `git branch`
**基础**：查看 / 创建 / 删除分支。  
**用法**：`git branch [选项] [分支名]`  
**案例**：
```bash
# 查看本地分支
git branch

# 查看所有分支（含远程）
git branch -a

# 创建新分支（不切换）
git branch feature/dice-loss

# 删除已合并分支
git branch -d feature/dice-loss

# 强制删除未合并分支
git branch -D feature/dice-loss
```

### `git checkout` / `git switch`
**基础**：切换分支或恢复文件。`git switch` 是更语义化的新版命令。  
**用法**：`git switch <分支名>`  
**案例**：
```bash
# 切换到已有分支
git switch main
git checkout main   # 旧写法，效果相同

# 创建并切换到新分支
git switch -c feature/focal-loss

# 恢复某个文件到上次提交状态（丢弃改动）
git checkout -- rsunet/engine.py
```

### `git merge`
**基础**：把指定分支合并进当前分支。  
**用法**：`git merge <分支名>`  
**案例**：
```bash
# 先切到 main
git switch main

# 把 feature 分支合并进来
git merge feature/dice-loss

# 只保留一个提交（压缩合并）
git merge --squash feature/dice-loss
```

---

## 6. 撤销与回退

### `git restore`
**基础**：恢复文件到指定状态（新版推荐，替代部分 checkout 场景）。  
**用法**：`git restore [--staged] <文件>`  
**案例**：
```bash
# 丢弃工作区对某文件的改动
git restore rsunet/engine.py

# 把已暂存的文件撤出暂存区（改动保留）
git restore --staged rsunet/engine.py
```

### `git reset`
**基础**：回退提交 / 暂存状态。三种模式区别关键。  
**用法**：`git reset [--soft | --mixed | --hard] <目标提交>`  
**案例**：
```bash
# 撤销最近一次提交，改动保留在暂存区
git reset --soft HEAD~1

# 撤销最近一次提交，改动保留在工作区（默认）
git reset HEAD~1

# ⚠️ 彻底丢弃最近一次提交及改动（危险）
git reset --hard HEAD~1

# 回退到指定 commit
git reset --hard abc123
```

> ⚠️ `--hard` 会**丢失未提交改动**，用前务必确认。

### `git revert`
**基础**：生成一个**反向提交**来撤销某次提交，**不改写历史**，适合已推送的场景。  
**用法**：`git revert <提交哈希>`  
**案例**：
```bash
# 撤销某次提交（安全，会新建一个 commit）
git revert abc123
```

---

## 7. 日志与历史

### `git log`
**基础**：查看提交历史。  
**用法**：`git log [选项]`  
**案例**：
```bash
# 完整历史
git log

# 单行简洁模式 + 图形化分支
git log --oneline --graph --all

# 查看某文件的修改历史
git log --oneline rsunet/engine.py

# 查看最近 5 条
git log -5 --oneline
```

### `git show` / `git blame`
**基础**：查看某次提交详情 / 逐行追溯修改者。  
**用法**：`git show <提交>` / `git blame <文件>`  
**案例**：
```bash
# 查看某次提交的具体改动
git show abc123

# 查看文件每一行是谁、哪次提交改的
git blame rsunet/metrics.py
```

---

## 8. 标签管理

### `git tag`
**基础**：为提交打标签，常用于标记版本发布。  
**用法**：`git tag [-a] <标签名>`  
**案例**：
```bash
# 轻量标签
git tag v1.0.0

# 附注标签（推荐，含说明信息）
git tag -a v1.0.0 -m "首个可复现版本"

# 推送标签到远程
git push origin v1.0.0
git push origin --tags   # 推送所有标签
```

---

## 9. 暂存工作区（stash）

### `git stash`
**基础**：临时保存未提交的改动，切分支 / 拉代码时不丢失。  
**用法**：`git stash [push/pop/list/drop]`  
**案例**：
```bash
# 暂存当前改动
git stash

# 暂存并加说明
git stash push -m "engine.py 半成品"

# 查看暂存列表
git stash list

# 恢复最近一次 stash 并删除记录
git stash pop

# 恢复但不删除记录
git stash apply

# 删除某条 stash
git stash drop stash@{0}
```

---

## 10. 冲突处理

**基础**：`git merge` / `git pull` / `git rebase` 遇到同一文件同一位置被不同修改时会产生冲突。

**冲突文件中的标记**：
```text
<<<<<<< HEAD
当前分支的内容
=======
要合并进来的分支内容
>>>>>>> feature/dice-loss
```

**处理流程**：
```bash
# 1. 查看冲突文件
git status

# 2. 手动编辑冲突文件，保留正确内容，删除 <<<< ==== >>>> 标记

# 3. 标记为已解决
git add rsunet/losses.py

# 4. 完成合并
git commit -m "解决 losses.py 合并冲突"
```

**放弃合并**：
```bash
git merge --abort
```

---

## 11. 常见工作流示例

### 🚀 日常提交并推送（最常用）
```bash
# 1. 看改了什么
git status
git diff

# 2. 加入暂存区（指定文件更安全）
git add rsunet/engine.py notes/day6_engine/

# 3. 提交到本地仓库
git commit -m "添加单轮训练引擎与 Day6 笔记"

# 4. 推送到 GitHub
git push origin main
```

### 🌿 新功能开发流程
```bash
# 1. 从 main 拉最新代码
git switch main
git pull origin main

# 2. 创建并切换到功能分支
git switch -c feature/dice-loss

# 3. 开发 → 提交
git add rsunet/losses.py
git commit -m "新增 Dice Loss 实现"

# 4. 推送分支到远程
git push -u origin feature/dice-loss

# 5. 在 GitHub 上发起 Pull Request，合并后删除分支
git switch main
git pull origin main
git branch -d feature/dice-loss
```

### 🆘 提交到错误分支的补救
```bash
# 场景：本该提交到 feature 分支，却提交到了 main（且未推送）

# 1. 记下当前提交
git log --oneline -1

# 2. 撤销 main 上的这次提交（改动保留）
git reset --soft HEAD~1

# 3. 切到正确分支
git switch feature/dice-loss

# 4. 重新提交
git commit -m "添加单轮训练引擎"
```

### 🧹 误把大文件（权重/数据）提交了
```bash
# 如果刚 add 但没 commit，直接撤出暂存区
git restore --staged runs/weights/best_miou.pth

# 如果已 commit 但没 push，撤销提交后重新提交
git reset --soft HEAD~1
git restore --staged runs/weights/best_miou.pth
echo "runs/" >> .gitignore
git add .gitignore
git commit -m "忽略 runs 目录，避免提交权重文件"
```

---

## 📌 速记口诀

| 场景 | 命令 |
| :--- | :--- |
| 初始化 | `git init` |
| 克隆 | `git clone <url>` |
| 看状态 | `git status` |
| 看改动 | `git diff` |
| 暂存 | `git add <file>` |
| 提交 | `git commit -m "msg"` |
| 推送 | `git push origin main` |
| 拉取 | `git pull origin main` |
| 切分支 | `git switch <branch>` |
| 建分支 | `git switch -c <branch>` |
| 看历史 | `git log --oneline --graph` |
| 撤暂存 | `git restore --staged <file>` |
| 撤提交 | `git reset --soft HEAD~1` |
| 安全撤销 | `git revert <commit>` |
| 临时存 | `git stash` / `git stash pop` |

---

> 💡 **核心原则**：  
> 1. **提交前先 `git status` + `git diff`**，别盲目 `add .`  
> 2. **提交信息写清楚"做了什么 + 为什么"**  
> 3. **已推送的历史尽量用 `revert` 而非 `reset --hard`**  
> 4. **危险命令（`--hard` / `--force`）用前三思**