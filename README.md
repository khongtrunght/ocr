**Create env**

```
conda create -n myenv python=3.7
conda activate myenv
pip install -r requirements.txt
```

**Train**

`!python ocr/tools/train.py -c ocr/my_config.yml \
     -o Global.checkpoints=path/to/checkpoint`

if you want to resume training from checkpoint, use `--checkpoint path/to/checkpoint --resume`

**Evaluate**

`python3 tools/eval.py -c my_config.yml  -o Global.checkpoints="path/to/checkpoint" PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5`


*Format of training file txt like original VietOCR*
