REMOTE_BASE_DIR=/home/user/border
LOCAL_BASE_DIR=$PWD/..

function copy_best_result() {
    SRC=$REMOTE_BASE_DIR/$1/best
    DST=$LOCAL_BASE_DIR/$1
    echo ====================================
    echo Copy result to $DST
    scp -r -i ~/.ssh/mykey.txt -P 20122 user@localhost:$SRC $DST
}

# DQN Pong
copy_best_result border/examples/atari/model/candle/dqn_pong
copy_best_result border/examples/atari/model/tch/dqn_pong

# SAC Ant
copy_best_result border/examples/ant/model/candle
copy_best_result border/examples/ant/model/tch
