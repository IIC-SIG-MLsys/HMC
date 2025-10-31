# HMC Developer Guide

This document explains how to contribute to the HMC project, including how to create branches, push code, keep your fork up to date, and format your code correctly.

---

## 🚀 How to Develop

### 1️⃣ Fork the Repository

* Go to the official GitHub repo:
  [https://github.com/IIC-SIG-MLsys/HMC](https://github.com/IIC-SIG-MLsys/HMC)
* Click **Fork** to create your own copy under your GitHub account.

---

### 2️⃣ Create a New Branch for Your Work

Always develop features or fixes in a **new branch** rather than directly on `main`.

Example branch naming conventions:

| Type     | Example                     |
| -------- | --------------------------- |
| Feature  | `feature/uhm-optimization`  |
| Bugfix   | `fix/rdma-recv-timeout`     |
| Refactor | `refactor/memory-interface` |
| Test     | `test/performance-bench`    |

---

### 3️⃣ Clone Your Fork and Checkout the Branch

```bash
# Clone your fork
git clone https://github.com/<your_username>/HMC.git
cd HMC

# Checkout a new branch from the remote branch
git checkout -b branch_name origin/branch_name

# After development
git add .

# Commit changes
git commit -m "your commit message" -s  # the -s adds a Signed-off-by line
```

If you want to modify your latest commit (for example, to combine commits or fix a message):

```bash
git commit --amend
```

Then push your branch to your fork:

```bash
git push origin branch_name
```

---

### 4️⃣ Create a Pull Request (PR)

1. Go to your forked repo on GitHub.
2. Click **“Compare & pull request”**.
3. Provide a **clear title** and **description** of what your PR does.
4. Request reviewers (if applicable).
5. Submit the PR to the main repository.

---

## 🧹 How to Format Code

### Install `clang-format`

Make sure you have it installed:

```bash
sudo apt-get install clang-format
```

### Run Formatting

> ⚠️ Note: Please **delete your build directory** (`build/`) before formatting to avoid unnecessary files being formatted.

Use the following command to format all C/C++ source files:

```bash
find . -type f -regex ".*\.\(cpp\|cc\|c\|h\|hpp\)$" -not -path "./extern/*" -print0 | xargs -0 clang-format -i
```

This automatically formats your code according to the project’s `.clang-format` style configuration.

---

## 🔄 How to Update Your Local Development Branch

When the upstream `main` branch updates, you can sync your fork to stay up to date.

### 1️⃣ Add the Upstream Remote

```bash
git remote add upstream https://github.com/IIC-SIG-MLsys/HMC
```

You can check remotes:

```bash
git remote -v
```

---

### 2️⃣ Fetch the Latest Changes

```bash
git fetch upstream
```

---

### 3️⃣ Merge Upstream to Your Local Branch

Option 1 (non-destructive merge):

```bash
git merge upstream/main --no-commit
```

Option 2 (preferred for linear history — **rebase**):

```bash
git rebase -i upstream/main
```

If conflicts appear, resolve them manually, then:

```bash
git add <conflicted_files>
git rebase --continue
```

Finally, push the updated branch to your fork:

```bash
git push --force
```

---

## 💡 Best Practices

✅ **Commit Guidelines**

* Keep commits small and self-contained.
* Write meaningful messages (e.g., `fix: correct RDMA buffer offset logic`).
* Use `-s` to sign your commits for DCO compliance.

✅ **Branch Guidelines**

* Always branch off `main`.
* Avoid long-lived branches; rebase frequently.
* Never commit build artifacts or large binaries.

✅ **PR Guidelines**

* Provide context and rationale in your PR description.
* Include test results or validation logs for major changes.
* Tag reviewers early for critical fixes.

---

## 🧾 Example Workflow Summary

```bash
# One-time setup
git clone https://github.com/<your_username>/HMC.git
cd HMC
git remote add upstream https://github.com/IIC-SIG-MLsys/HMC

# Create a new branch for feature development
git checkout -b feature/new-interface

# Make your changes
vim src/example.cpp
git add .
git commit -m "feat: add unified memory test interface" -s
git push origin feature/new-interface

# Sync with upstream later
git fetch upstream
git rebase upstream/main
git push --force

# Finally, open a PR from your fork to upstream/main
```

---

## 🧭 Summary

* Fork → Branch → Develop → Commit → PR
* Keep your code formatted with `clang-format`.
* Rebase frequently to avoid merge conflicts.
* Write clear commits and PR descriptions.

---

```
© 2025 SDU spgroup Holding Limited  
All Rights Reserved.
```