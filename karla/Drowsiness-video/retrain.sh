# python retrain/retrain.py \
    # --bottleneck_dir bottleneck \
    # --how_many_training_steps 200 \
    # --model_dir inception_model/ \
    # --output_graph output_graph.pb \
    # --output_labels output_labels.txt \
    # --image_dir retrain/data/train
#
python retrain/retrain_mobile.py --bottleneck_dir bottleneck --how_many_training_steps 4000 --output_graph output_graph.pb --output_labels output_labels.txt --image_dir retrain/data/train --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/feature_vector/3 --logdir retrain_logs