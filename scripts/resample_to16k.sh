#!/usr/bin/env bash
set -euo pipefail

# 1) 读取参数并规范成绝对路径
SRC="${1:?usage: $0 SRC_DIR DST_DIR}"
DST="${2:?usage: $0 SRC_DIR DST_DIR}"
SRC="$(realpath -e "$SRC")"
DST="$(realpath -m "$DST")"

# 2) 找到 ffmpeg 绝对路径
FFMPEG="$(command -v ffmpeg || true)"
FFPROBE="$(command -v ffprobe || true)"
if [[ -z "$FFMPEG" || -z "$FFPROBE" ]]; then
  echo "[ERR] ffmpeg/ffprobe not found. conda activate sec-gpu && mamba install -c conda-forge ffmpeg" >&2
  exit 1
fi

# 3) 防止 SRC 和 DST 指到同一位置
if [[ "$SRC" -ef "$DST" ]]; then
  echo "[ERR] SRC and DST are the same directory: $SRC" >&2
  exit 1
fi

export SRC DST FFMPEG FFPROBE
find "$SRC" -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' -o -iname '*.m4a' \) -print0 |
  xargs -0 -I{} -P 8 bash -c '
    in="$1"
    rel="${in#"$SRC"/}"
    out="$DST/${rel%.*}.wav"
    mkdir -p "$(dirname "$out")"

    # 如果已是16k/单声道/s16le且扩展名为wav，就跳过
    sr="$("$FFPROBE" -v error -select_streams a:0 \
          -show_entries stream=sample_rate,channels,codec_name \
          -of default=nk=1:nw=1 "$in" 2>/dev/null | paste -sd, -)"
    if [[ "${sr}" == "16000,1,pcm_s16le" && "${in##*.}" == "wav" ]]; then
      [[ -e "$out" ]] || cp -n "$in" "$out"
      echo "[SKIP] already 16k mono s16le: $rel"
      exit 0
    fi

    "$FFMPEG" -hide_banner -loglevel error -y \
      -i "$in" -vn -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le "$out"
    echo "[OK] $out"
  ' _ {}

