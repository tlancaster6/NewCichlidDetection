# NewCichlidDetection

Base train command:

`python train.py --img 1296 --cfg yolov5s.yaml --hyp hyp.scratch.yaml --batch 32 --epochs 100 --data cichlid.yaml --weights yolov5s.pt --workers 24 --name yolo_cichlid`

Add `--device 0` to train on GPU.