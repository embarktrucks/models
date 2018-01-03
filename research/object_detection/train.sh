PATH_TO_YOUR_PIPELINE_CONFIG=object_detection/samples/configs/faster_rcnn_resnet101_pets.config
PATH_TO_BASE_DIR=/sbox/users/blake/models/perception/objects/faster_rcnn/
FOLDER_EXT=`find ${LOG_DIR} -maxdepth 1 -type d -regextype posix-extended -regex "${PATH_TO_BASE_DIR}/[0-9]{3}" | wc -l| printf "%03d"`

PATH_TO_TRAIN_DIR="${PATH_TO_BASE_DIR}/${FOLDER_EXT}/train"
PATH_TO_EVAL_DIR="${PATH_TO_BASE_DIR}/${FOLDER_EXT}/test"
# mkdir -p ${PATH_TO_TRAIN_DIR}
# mkdir -p ${PATH_TO_EVAL_DIR}

COMMAND="python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR}"
echo $COMMAND    

COMMAND="python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}"
echo $COMMAND
eval $COMMAND