export MLFLOW_TRACKING_URI=http://localhost:8080

import-experiment --experiment-name Gym --input-dir export/Gym
import-experiment --experiment-name Atari --input-dir export/Atari
