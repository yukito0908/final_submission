#!/usr/bin/env bash
set -euo pipefail

ign service -s /world/default/create \
  --reqtype ignition.msgs.EntityFactory \
  --reptype ignition.msgs.Boolean \
  --timeout 5000 \
  --req 'sdf_filename: "/workspace/assets/pick_place_markers.sdf" name: "pick_place_markers" allow_renaming: false'
