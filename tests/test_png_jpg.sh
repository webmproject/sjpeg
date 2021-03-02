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

# tests for JPEG/PNG/PPM -> JPEG -> JPEG chain

SJPEG=../examples/sjpeg
TMP_JPEG1=/tmp/test1.jpg
TMP_JPEG2=/tmp/test2.jpg

LIST="source1.png \
      source1.itl.png \
      source2.jpg \
      source3.jpg \
      source4.ppm \
      test_icc.jpg \
      test_exif_xmp.png"

set -e
for f in ${LIST}; do
  ${SJPEG} testdata/${f} -o ${TMP_JPEG2} -info -q 56.7 -no_limit
  ${SJPEG} testdata/${f} -o ${TMP_JPEG2} -size 16000 -pass 3 -yuv_mode 4
done

for f in ${LIST}; do
  ${SJPEG} testdata/${f} -o ${TMP_JPEG1} -quiet -psnr 39
  ${SJPEG} ${TMP_JPEG1} -o ${TMP_JPEG2} -r 88.7 -short -info -size 20000
done

for f in ${LIST}; do
  ${SJPEG} testdata/${f} -o ${TMP_JPEG1} -quiet -no_metadata
  ${SJPEG} ${TMP_JPEG1} -r 76.6542 -o ${TMP_JPEG2} -short
done

echo "OK!"
exit 0
