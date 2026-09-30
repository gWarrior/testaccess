#!/usr/bin/env bash
# Download the training corpus: 5 shards (~1.26 GB parquet, ~2.4 GB text) of
# Taiga "proza" (cointegrated/taiga_stripped_proza, CC BY-SA 3.0), verify them
# and place them where `snn-lm prepare` looks by default.
set -euo pipefail
DEST="${1:-/home/user/data/taiga}"
BASE="https://huggingface.co/datasets/cointegrated/taiga_stripped_proza/resolve/main/data"
mkdir -p "$DEST/extra"
cd "$DEST"
while read -r sum file; do
  [ -f "$file" ] || curl -fL --retry 4 -o "$file" "$BASE/$file"
  echo "$sum  $file" | sha256sum -c -
done <<'LIST'
e6af8512464599a417b12e1a142660c52804d9151b8f59db76d71f604ce565cb train-00000-of-00083-5a836a36820bbc21.parquet
ea058dc4cf11b5035f0b9d1dfe377234209f72275fa726641917d877da2415fa train-00001-of-00083-6a059492052de562.parquet
604b0eb2dce7dff68db0240585f5b3769bb5ac0d232ce0022db0a0f662b0d995 train-00002-of-00083-6ab99ef2eda1556f.parquet
2d0b00927be0bf5293593d57894204fd2a4bc6b166ce977b28e4703651dc2ceb train-00003-of-00083-fc34df8e6a0b97a4.parquet
54e855a8833f32b97799ec31e8bf588be0034d7a68079b4f071241324e60deca train-00004-of-00083-a2a0fa5d28e7d578.parquet
LIST
echo "Corpus ready in $DEST. Put extra UTF-8 .txt files (your own books) into $DEST/extra/."
