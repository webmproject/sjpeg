#!/bin/sh
# tests for command lines

SJPEG=../examples/sjpeg
TMP_FILE1=/tmp/test.jpg
TMP_FILE2=/tmp/test
BAD_FILE=/tmp/
SRC_FILE1="./testdata/source1.png"
SRC_FILE2="./testdata/source2.jpg"
SRC_FILE3="./testdata/source4.ppm"

# simple coverage of command line arguments. Positive tests.
echo "POSITIVE TESTS"
set -e
${SJPEG} -version
${SJPEG} -h
${SJPEG} --help

${SJPEG} ${SRC_FILE1} -o ${TMP_FILE1} -yuv_mode 2
${SJPEG} ${SRC_FILE1} -o ${TMP_FILE1} -q 3 -no_adapt -no_optim -quiet

${SJPEG} ${SRC_FILE1} -o ${TMP_FILE1} -r 30 -no_adapt -no_optim -quiet
${SJPEG} ${SRC_FILE2} -o ${TMP_FILE1} -r 30 -no_adapt -no_optim -quiet
${SJPEG} ${SRC_FILE3} -o ${TMP_FILE1} -r 30 -no_adapt -no_optim -quiet

# negative tests (should fail)
echo "NEGATIVE TESTS"
set +e
${SJPEG} -no_adapt -no_optim -quiet
${SJPEG} ${SRC_FILE1} -o ${TMP_FILE1} -yuv_mode -1 -quiet
${SJPEG} ${SRC_FILE1} -o ${TMP_FILE1} -yuv_mode 99 -quiet -no_metadata
${SJPEG} -q 80 -quiet
${SJPEG} ${SRC_FILE1} -risk -quiet
${SJPEG} ${SRC_FILE1} -o
${SJPEG} -o ${TMP_FILE2} -quiet
${SJPEG} ${SRC_FILE1} -o ${TMP_FILE1} -yuv_mode -1 -quiet
${SJPEG} ${SRC_FILE1} -o ${TMP_FILE1} -yuv_mode 99 -quiet
${SJPEG} ${BAD_FILE} -o ${TMP_FILE1} -quiet
${SJPEG} ${SRC_FILE1} -o ${BAD_FILE} -quiet

${SJPEG} ${SRC_FILE2} -o
${SJPEG} -o ${BAD_FILE} -quiet

echo "OK!"
exit 0
