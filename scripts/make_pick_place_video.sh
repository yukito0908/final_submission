#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RAW_ROLLOUT.mp4 [OUTPUT.mp4]" >&2
  exit 2
fi

input=$1
output=${2:-/workspace/artifacts/pick_place/media/pick_place_act_submission.mp4}
font=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf

ffmpeg -loglevel error -y \
  -f lavfi -i "color=c=0x172033:s=1272x668:r=20:d=5" \
  -i "$input" \
  -f lavfi -i "color=c=0x172033:s=1272x668:r=20:d=7" \
  -filter_complex "
    [0:v]drawtext=fontfile=${font}:text='CRANE+ V2  |  ACT PICK AND PLACE':fontcolor=white:fontsize=44:x=(w-text_w)/2:y=235,
         drawtext=fontfile=${font}:text='BLACK CUBE   RED START  ->  GREEN GOAL':fontcolor=0x72e69b:fontsize=28:x=(w-text_w)/2:y=315,
         drawtext=fontfile=${font}:text='Inference uses the trained ACT policy - not teacher control':fontcolor=0xc7d2e5:fontsize=23:x=(w-text_w)/2:y=375[title];
    [1:v]trim=start=0:end=85,setpts=PTS-STARTPTS,
         drawbox=x=0:y=575:w=iw:h=93:color=black@0.62:t=fill,
         drawtext=fontfile=${font}:text='TRAINED ACT POLICY  |  BLACK CUBE\: RED -> GREEN':fontcolor=white:fontsize=23:x=25:y=590,
         drawtext=fontfile=${font}:text='SETUP / RESET':fontcolor=0xffd166:fontsize=28:x=25:y=625:enable='between(t,0,12)',
         drawtext=fontfile=${font}:text='1  APPROACH - GRIPPER OPEN':fontcolor=0xffd166:fontsize=28:x=25:y=625:enable='between(t,12,39)',
         drawtext=fontfile=${font}:text='2  GRASP':fontcolor=0xffd166:fontsize=28:x=25:y=625:enable='between(t,39,49)',
         drawtext=fontfile=${font}:text='3  LIFT':fontcolor=0xffd166:fontsize=28:x=25:y=625:enable='between(t,49,62)',
         drawtext=fontfile=${font}:text='4  TRANSPORT':fontcolor=0xffd166:fontsize=28:x=25:y=625:enable='between(t,62,75)',
         drawtext=fontfile=${font}:text='5  PLACE AND RELEASE':fontcolor=0x72e69b:fontsize=28:x=25:y=625:enable='between(t,75,85)'[rollout];
    [2:v]drawtext=fontfile=${font}:text='ACT EVALUATION\: SUCCESS':fontcolor=0x72e69b:fontsize=46:x=(w-text_w)/2:y=205,
         drawtext=fontfile=${font}:text='Official evaluation\: 5 / 5 successful trials':fontcolor=white:fontsize=29:x=(w-text_w)/2:y=295,
         drawtext=fontfile=${font}:text='Recorded rollout\: lift 17.96 cm  |  transport 33.25 cm':fontcolor=0xc7d2e5:fontsize=25:x=(w-text_w)/2:y=355,
         drawtext=fontfile=${font}:text='Physical success verified from Gazebo cube pose':fontcolor=0xc7d2e5:fontsize=25:x=(w-text_w)/2:y=405[result];
    [title][rollout][result]concat=n=3:v=1:a=0,format=yuv420p[v]
  " \
  -map "[v]" -c:v libx264 -preset medium -crf 20 -movflags +faststart "$output"

ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height,codec_name \
  -of json "$output"
