#!/bin/sh
# models/ is bind-mounted from the host, so the first start of a checkout fetches
# what is missing into it and every start after that finds the files and does
# nothing. Fetching here rather than in the build keeps the image small and lets
# a checkout that already carries its own (fine-tuned) weights stay offline.
# Failing here beats every capture answering "model weights missing".
set -e
mkdir -p /app/models
for name in yolo26s-seg.pt yolo26s-pose.pt; do
    if [ ! -f "/app/models/$name" ]; then
        echo "fetching $name into /app/models"
        (cd /app/models && python -c "from ultralytics import YOLO; YOLO('$name')")
    fi
done
exec "$@"