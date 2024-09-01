# Export
ssh -i ~/.ssh/mykey.txt -p 20122 user@localhost 'mkdir ~/export'
ssh -i ~/.ssh/mykey.txt -p 20122 user@localhost \
    'echo "
export MLFLOW_TRACKING_URI=http://localhost:8080
/home/user/venv/bin/export-experiment --experiment Gym --output-dir /home/user/export/Gym
#/home/user/venv/bin/export-experiment --experiment Atari --output-dir /home/user/export/Atari
" > tmp.sh'

ssh -i ~/.ssh/mykey.txt -p 20122 user@localhost 'bash tmp.sh'

# Remote copy
rm -fr $PWD/export
scp -r -i ~/.ssh/mykey.txt -P 20122 user@localhost:/home/user/export $PWD
