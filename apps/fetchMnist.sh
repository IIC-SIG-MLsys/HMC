#!/bin/bash

DATA_DIR="${PWD}/dataset"
MNIST_DIR="${DATA_DIR}/mnist"
MNIST_URL="https://ossci-datasets.s3.amazonaws.com/mnist"

mkdir -p "${MNIST_DIR}"

declare -A MNIST_EXTRACTED_FILES_MD5=(
    ["t10k-images-idx3-ubyte"]="2646ac647ad5339dbf082846283269ea"
    ["t10k-labels-idx1-ubyte"]="27ae3e4e09519cfbb04c329615203637"
    ["train-images-idx3-ubyte"]="6bbc9ace898e44ae57da46a324031adb"
    ["train-labels-idx1-ubyte"]="a25bea736e30d166cdddb491f175f624"
)

declare -A MNIST_ARCHIVE_MD5=(
    ["t10k-images-idx3-ubyte"]="9fb629c4189551a2d022fa330f9573f3"
    ["t10k-labels-idx1-ubyte"]="ec29112dd5afa0611ce80d1b7f02629c"
    ["train-images-idx3-ubyte"]="f68b3c2dcbeaaa9fbdd348bbdeb94873"
    ["train-labels-idx1-ubyte"]="d53e105ee54ea40749a09fcbcd1e9432"
)

check_files() {
    local -n files=$1
    local -n md5s=$2
    local missing=()

    for file in "${!files[@]}"; do
        if [ ! -f "${MNIST_DIR}/${file}" ] || ! (echo "${md5s[$file]}  ${MNIST_DIR}/${file}" | md5sum --check -); then
            missing+=("${file}")
        fi
    done

    echo "${missing[@]}"
}

MISSING_FILES=$(check_files MNIST_EXTRACTED_FILES_MD5 MNIST_ARCHIVE_MD5)

if [ -n "$MISSING_FILES" ]; then
    echo "Fetching MNIST dataset..."

    for FILE_TO_FETCH in $MISSING_FILES; do
        EXPECTED_FILE_MD5=${MNIST_ARCHIVE_MD5["$FILE_TO_FETCH"]}

        wget -O "${MNIST_DIR}/${FILE_TO_FETCH}.gz" --no-check-certificate "${MNIST_URL}/${FILE_TO_FETCH}.gz"

        if ! (echo "${EXPECTED_FILE_MD5}  ${MNIST_DIR}/${FILE_TO_FETCH}.gz" | md5sum --check -); then
            echo "MD5 checksum failed for ${FILE_TO_FETCH}.gz"
            exit 1
        fi

        gunzip -c "${MNIST_DIR}/${FILE_TO_FETCH}.gz" > "${MNIST_DIR}/${FILE_TO_FETCH}"
        rm "${MNIST_DIR}/${FILE_TO_FETCH}.gz"
    done

    echo "Fetching MNIST dataset - done"
fi