#!/bin/sh
set -e
DEST=/data/mobile_sam.pt
if [ -f "$DEST" ]; then
    echo "checkpoint already present, skipping download"
    exit 0
fi
URL="https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
echo "downloading $URL ..."
python3 -c "import urllib.request; urllib.request.urlretrieve('$URL', '$DEST')"
echo "done"
