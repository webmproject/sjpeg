#!/bin/sh
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

# tests for command lines

SJPEG=../examples/sjpeg
TMP_FILE1=/tmp/test.jpg
TMP_FILE2=/tmp/test
BAD_FILE=/tmp/
SRC_FILE1="./testdata/source1.png"
SRC_FILE2="./testdata/source2.jpg"
SRC_FILE3="./testdata/source3.jpg"   # large file
SRC_FILE4="./testdata/source4.ppm"

# simple coverage of command line arguments. Positive tests.
echo
echo "=== POSITIVE TESTS ==="
echo

set -e
${SJPEG} -version
${SJPEG} -h
${SJPEG} --help

${SJPEG} ${SRC_FILE1} -o ${TMP_FILE1} -yuv_mode 2
${SJPEG} ${SRC_FILE1} -o ${TMP_FILE1} -yuv_mode 3
${SJPEG} ${SRC_FILE1} -o ${TMP_FILE1} -q 3 -no_adapt -no_optim -quiet -sharp

${SJPEG} ${SRC_FILE1} -o ${TMP_FILE1} -r 30 -no_adapt -no_optim -quiet -420
${SJPEG} ${SRC_FILE2} -o ${TMP_FILE1} -q 24 -psnr 35 -pass 5 \
                                      -trellis -adapt_bias -quiet
${SJPEG} ${SRC_FILE4} -o ${TMP_FILE1} -size 24000 -pass -1 -tolerance .2 \
                                      -444 -quiet
${SJPEG} ${SRC_FILE1} -crc
${SJPEG} ${SRC_FILE1} -estimate
${SJPEG} ${SRC_FILE1} -i

# test CRC is matching
if [ -x "$(command -v md5)" ]; then
  for file in ${SRC_FILE1} ${SRC_FILE2} ${SRC_FILE4}; do
    ${SJPEG} ${file} -o ${TMP_FILE1} -quiet
    ${SJPEG} ${file} -md5
    md5 ${TMP_FILE1}
  done
else
  echo "'md5' command is not available. Skipping MD5 test."
fi

# test -xmp / -exif / -icc
echo "This is a test. We need a looooooooooooong line" > ${TMP_FILE1}
${SJPEG} ${SRC_FILE1} -xmp ${TMP_FILE1} -exif ${TMP_FILE1} -icc ${TMP_FILE1}
echo "LARGE ICC" && ${SJPEG} ${SRC_FILE1} -icc ${SRC_FILE3} \
                             -quiet -o ${TMP_FILE1}
${SJPEG} ${TMP_FILE1} -o ${TMP_FILE1} -r 76

# negative tests (should fail)
echo
echo "=== NEGATIVE TESTS ==="
echo

set +e
${SJPEG}
${SJPEG} -no_adapt -no_optim -quiet
${SJPEG} ${SRC_FILE1} -o ${TMP_FILE1} -yuv_mode -1 -quiet
${SJPEG} ${SRC_FILE1} -o ${TMP_FILE1} -yuv_mode 4 -quiet
${SJPEG} ${SRC_FILE1} -o ${TMP_FILE1} -yuv_mode 99 -quiet -no_metadata
${SJPEG} -q 80 -quiet
${SJPEG} ${SRC_FILE1} -risk -quiet
${SJPEG} ${SRC_FILE1} -o
${SJPEG} -o ${TMP_FILE2} -quiet
${SJPEG} ${SRC_FILE1} -r 101
${SJPEG} ${SRC_FILE1} -r -1
${SJPEG} ${BAD_FILE} -o ${TMP_FILE1} -quiet
${SJPEG} ${SRC_FILE1} -o ${BAD_FILE} -quiet

${SJPEG} ${SRC_FILE2} -o
${SJPEG} -o ${BAD_FILE} -quiet

# test with large EXIF
echo "LARGE EXIF" && ${SJPEG} ${SRC_FILE1} -exif ${SRC_FILE3} -quiet
# test with large file (>64kb). XMP can't handle larger-than-64k data
echo "LARGE XMP" && ${SJPEG} ${SRC_FILE1} -xmp ${SRC_FILE3} -quiet -o ${TMP_FILE1}

# this test does not work for very low quality values (q<4)
for q in `seq 4 100`; do
  ${SJPEG} -q $q ${SRC_FILE1} -o ${TMP_FILE1} -no_adapt -no_optim &> /dev/null
  # parse the 'estimated quality' result string, and compare to expected quality
  a=(`${SJPEG} -i ${TMP_FILE1} | grep estimated | grep -Eo '[+-]?[0-9]+(\.0)'`)
  q1="${a[0]}"
  q2="${a[1]}"
  q3="${q}.0"
  if [ "x${q1}" != "x${q3}" ]; then echo "Y-Quality mismatch!"; exit 1; fi
  if [ "x${q2}" != "x${q3}" ]; then echo "UV-Quality mismatch!"; exit 1; fi
done

echo "OK!"
exit 0
