# How to dev?

1. fork your own repo
2. create new branch for special development
3. clone repo:branch to local env
    ```
    git clone url

    git checkout -b branch_name origin branch_name

    # after developing
    git add .

    git commit -m "msg" -s

    git commit --amend # if merge commit to the last commit

    git push
    ```
4. create a PR on github

# How to format code
> you need install clang-format first

> note: make sure you have deleted the build dir

```
find . -type f -regex ".*\.\(cpp\|cc\|c\|h\|hpp\|cu\|cuh\|py\|mlu\|hip\)$" -not -path "./extern/*" -print0 | xargs -0 clang-format -i
``` 

# How to update local dev branch from upstream
1. add upstream
    ```
    git remote add upstream https://github.com/IIC-SIG-MLsys/HDDT
    ```
2. fetch
    ```
    git fetch upstream
    ```
3. merge upstream to local
    ```
    git merge upstream/main --no-commit
    ```
4. rebase
    ```
    git rebase -i upstream/main
    ```
