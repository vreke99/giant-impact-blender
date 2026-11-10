#!/bin/bash
#
#   Stage 3: render frames START-END from a template, running MAX_PARALLEL
#   Blender processes at once on this machine's GPU.
#       ./render_chunks.sh                 # frames 0-2160 (or as set below)
#       ./render_chunks.sh 300 599         # a sub-range
#   stage3_slurm.sh runs this on a chunk of the range on each GPU node it is given.
#   MAX_PARALLEL and NTHREAD can also be set in the environment (as stage3_slurm.sh does).

START_FRAME=${1:-0}
END_FRAME=${2:-2160}
CHUNK=50
MAX_PARALLEL=${MAX_PARALLEL:-8}  # concurrent Blender processes: limited by GPU memory
NTHREAD=${NTHREAD:-12}           # threads each: MAX_PARALLEL x NTHREAD <= cores (128 on mad06)

(( CHUNK > (END_FRAME - START_FRAME + MAX_PARALLEL) / MAX_PARALLEL )) &&
    CHUNK=$(( (END_FRAME - START_FRAME + MAX_PARALLEL) / MAX_PARALLEL ))   # keep every worker busy

# Directory setup
BLENDER_BIN="/cosma/local/software/blender/blender-git/blender/my_mad06_build/bin/blender"
OUT_DIR="/cosma5/data/dp004/vreke/blender/outs_and_errs"
mkdir -p "$OUT_DIR"
SIM_NAME="pE65_Th35_b60v10_3e6"
MODE="mesh_mist"         # lower-case RENDER_MODE: particles, mesh_particles, mesh_mist, particles_mist
CAMERA="vr360_3d"           # lower-case CAMERA_MODE: persp, ortho, fisheye, vr360 (add _3d for stereo)
BLEND_FILE="/cosma5/data/dp004/vreke/swift/rssse/${SIM_NAME}/${SIM_NAME}_${MODE}_${CAMERA}.blend"
RENDER_SCRIPT="/cosma/home/planet/vreke/blender/combined_03_render_chunk.py"

if [ ! -f "$BLEND_FILE" ]; then
    echo "Template blend not found:"
    echo "  $BLEND_FILE"
    exit 1
fi
if [ ! -f "$RENDER_SCRIPT" ]; then
    echo "Render script not found:"
    echo "  $RENDER_SCRIPT"
    exit 1
fi

# Calculate total chunks upfront for clarity
TOTAL_CHUNKS=$(( (END_FRAME - START_FRAME + CHUNK) / CHUNK ))
echo "=================================================="
echo "Range: Frames ${START_FRAME} to ${END_FRAME}"
echo "Chunk Size: ${CHUNK} frames | Max Parallel Workers: ${MAX_PARALLEL}"
echo "Total Chunks to process: ${TOTAL_CHUNKS}"
echo "=================================================="

pids=(); ranges=()
for (( curr_start=$START_FRAME; curr_start<=$END_FRAME; curr_start+=$CHUNK )); do
  curr_end=$((curr_start + CHUNK - 1))
  [ $curr_end -gt $END_FRAME ] && curr_end=$END_FRAME

  log="${OUT_DIR}/render_${curr_start}_to_${curr_end}.log"

  # --- SAFETY GUARD: Throttle background processes ---
  while true; do
    # Count how many of OUR blender processes are currently running
    active_jobs=$(jobs -r | wc -l)
    
    if [ "$active_jobs" -lt "$MAX_PARALLEL" ]; then
      break
    fi
    # If at capacity, wait 3 seconds before checking again
    sleep 3
  done
  # ---------------------------------------------------

  active_jobs=$(jobs -r | wc -l)
  echo "[Running: $((active_jobs + 1))/${MAX_PARALLEL}] Launching frames ${curr_start} to ${curr_end} -> ${log}"

  # Launch process in background
  "$BLENDER_BIN" --factory-startup -t ${NTHREAD} -b "$BLEND_FILE" -P "$RENDER_SCRIPT" -- $curr_start $curr_end > "$log" 2>&1 &
  pids+=($!); ranges+=("${curr_start} ${curr_end}")

  # Brief stagger between launches
  sleep 2
done

echo "--------------------------------------------------"
echo "All chunks submitted. Waiting for active workers to complete..."
failed=0
for k in "${!pids[@]}"; do
  if ! wait "${pids[$k]}"; then
    failed=1
    echo "FAILED: frames ${ranges[$k]}  (see ${OUT_DIR}/render_${ranges[$k]// /_to_}.log)"
  fi
done
if [ $failed -eq 0 ]; then
  echo "All frames from ${START_FRAME} to ${END_FRAME} complete!"
else
  echo "Some chunks failed: rerun just those ranges."
  exit 1
fi
