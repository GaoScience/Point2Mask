
# Point2Mask

Official implementation of PAMG and RPR-Net from the paper *Point-to-Mask: A
Two-Stage Framework for Infrared Small Target Detection with Arbitrary Point
Supervision*, for infrared small target detection under arbitrary point
supervision. If you find the paper, dataset, or code useful, please consider
starring this repository. The authors sincerely appreciate your support.

> **Research-use notice:** The code and released weights are available only for
> non-commercial academic research and education. Commercial use requires a
> separate written license. See [LICENSE](LICENSE).

RPR-Net uses Temporal Difference Attention (TDA) to predict target centers and radii from temporal infrared frames. The predictions can be converted into masks in two ways:

- `radius`: draw a radius mask directly from each predicted center and radius.
- `pamg`: use each predicted center as the PAMG seed and convert its predicted radius into the PAMG `Rs` growth prior.

This repository also includes Label-IRST, an interactive annotation tool with PAMG-assisted mask generation.

[中文说明](README_CN.md)

## Paper

**W. Gao et al., "Point-to-Mask: A Two-Stage Framework for Infrared Small Target Detection with Arbitrary Point Supervision," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2026.3730901.**

[[IEEE Xplore](https://ieeexplore.ieee.org/document/11683233/)] [[DOI](https://doi.org/10.1109/TGRS.2026.3730901)]

## Demos

### Label-IRST annotation and PAMG-assisted mask generation

<p align="center">
  <img src="assets/label_irst_demo.gif" alt="Label-IRST annotation and PAMG-assisted mask generation" width="900">
</p>

### Synchronized sequence comparison

<p align="center">
  <img src="assets/rprnet_comparison.gif" alt="Input sequence, RPR-Net detection, and RPR-Net plus PAMG comparison" width="960">
</p>

## Released components

- PAMG point-to-mask algorithm
- RPR-Net model and loss
- Supervised training script
- Single-sequence and dataset inference
- Radius and RPR-Net + PAMG mask generation
- Label-IRST annotation tool
- Official model weights trained on SIRSTD
- Demonstrations of Label-IRST, RPR-Net, and PAMG

## Installation

Python 3.9 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## Official weights

| Model | File | SHA-256 |
|---|---|---|
| RPR-Net on SIRSTD | `weights/rprnet_sirstd_v1.pth` | `D3CC256E02B93B56569A7BE604886573928CEAD3C6482F6801DAE29AAA333F32` |

## Quick inference

Provide three ordered frames centered on the target frame.

Radius-mask mode:

```bash
python RPRNet/tools/inference_rprnet.py \
  --images frame_0001.png frame_0002.png frame_0003.png \
  --mode radius \
  --output-dir outputs/radius
```

RPR-Net + PAMG mode:

```bash
python RPRNet/tools/inference_rprnet.py \
  --images frame_0001.png frame_0002.png frame_0003.png \
  --mode pamg \
  --output-dir outputs/pamg
```

Each run saves `detections.csv` and the generated binary mask.

To infer a dataset test split without reading reference masks:

```bash
python RPRNet/tools/inference_rprnet.py \
  --sample-list dataset/SIRSTD/splits/test.txt \
  --image-root dataset/SIRSTD/images \
  --mode pamg \
  --output-dir outputs/test_predictions
```

Sample IDs must follow the SIRSTD naming pattern, for example `IR_2_00000`.

## Minimal training

RPR-Net reads binary masks and builds its heatmap and regression supervision during training:

Set the model, label-generation, loss, and optimization hyperparameters in
`RPRNet/configs/rprnet_sirstd.py`, then run:

```bash
python -u RPRNet/tools/train_rprnet.py \
  --train-list dataset/SIRSTD/splits/train.txt \
  --image-root dataset/SIRSTD/images \
  --mask-root dataset/SIRSTD/masks \
  --save-dir outputs/rprnet_train
```

The training list contains one SIRSTD-style sample ID per line, without an
absolute path. Images and masks use the same ID, for example
`images/IR_2_00000.jpg` and `masks/IR_2_00000.png`.
`checkpoint_last.pth` contains the model, optimizer, learning-rate scheduler,
and epoch for resuming training; files named `rprnet_epoch_XXXX.pth` contain
model parameters only and are intended for inference.

Resume training with:

```bash
python -u RPRNet/tools/train_rprnet.py \
  --train-list dataset/SIRSTD/splits/train.txt \
  --image-root dataset/SIRSTD/images \
  --mask-root dataset/SIRSTD/masks \
  --save-dir outputs/rprnet_train \
  --resume outputs/rprnet_train/checkpoint_last.pth
```

The inference script does not read the training configuration. Model, batch
size, threshold, Top-K, raw-output saving, PAMG, and other inference parameters
are grouped in the `Editable inference settings` section near the top of
`RPRNet/tools/inference_rprnet.py`, so the script can be edited and run
independently. PAMG inference settings include `PAMG_RS_SCALE`, `PAMG_RS_CAP`,
and `PAMG_POLARITY`. Dataset inference reads only images and sample lists; it
does not read ground-truth masks.

## Label-IRST

```bash
python labelSIRST/sirst_mask_labeler.py
```

Label-IRST consists of an Overview pane on the left and a Detail View pane on
the right. Use the Overview pane to browse images and select target regions. Use
the Detail View to run PAMG and add to or erase from the mask. The crosshairs in
the two panes are synchronized and show the current pixel coordinates and
grayscale value.

If you think the tool could be improved, suggestions and changes are welcome.
The current version is operated as follows:

### Workspace directories

Configure the following directories when the application starts:

- **Images Dir**: directory containing the images to annotate. Supported formats
  are `.jpg`, `.png`, `.bmp`, and `.tif`.
- **Masks Dir**: directory from which masks are loaded and to which they are
  saved. An existing mask with the same base name is loaded automatically;
  otherwise, annotation starts with an empty mask. Masks are always saved as
  `.png` files with the same base name as the image. Do not use the image
  directory as the mask directory, because this may overwrite input images.
- **Labels Dir (Optional)**: optional directory containing YOLO bounding-box
  annotations. Each image uses a `.txt` file with the same base name. The tool
  reads the normalized center coordinates, width, and height from standard YOLO
  annotations to initialize the local editing regions. If the directory is
  empty or an image has no corresponding annotation, create the region manually
  in the Overview pane.

The directory settings are stored in `labeler_config.json` in the current
working directory and loaded automatically the next time the tool starts. This
file is ignored by Git. Use **Workspace Config** to select different directories
during a session; save the current mask before reconfiguring the workspace.

### Basic workflow

1. Select an image from the file list on the left. If YOLO annotations are
   available, the tool creates target boxes automatically. Otherwise, hold
   `Ctrl` and drag with the left mouse button in the Overview pane to create a
   target region.
2. Click a target box to select it and display its enlarged region in the Detail
   View. Press `Tab` to cycle through multiple target boxes.
3. Set `Rs` and the target-polarity mode under **Seed Grow**, then click
   **Apply Algo** to activate the settings.
4. In the Detail View, hold `Shift` and click the target point. PAMG generates
   an initial mask and merges it into the current mask.
5. Use the left-button brush to add foreground and the right-button brush to
   erase incorrect regions. Press `Ctrl+S` when finished.

The current mask is also saved automatically when switching images or closing
the application. In the file list, `✅` indicates a saved non-empty mask, `⚠️`
indicates a saved empty mask, `⬜` indicates that no corresponding mask file
exists, and `❓` indicates that the status is still being checked.

### Mouse controls

| Pane | Mouse action | Function |
|---|---|---|
| Overview | Left-click a target box | Select the box and display its region in the Detail View |
| Overview | Left-button drag | Pan the full image |
| Overview | `Ctrl + left-button drag` on empty space | Create a target box |
| Overview | `Ctrl + left-button drag` inside a target box | Move the target box |
| Overview | `Ctrl + left-button drag` on a box edge or handle | Resize the target box |
| Overview | Right-click a target box | Open the menu and select **Delete BBox** to remove it |
| Overview | `Ctrl + mouse wheel` | Zoom the full image around the pointer |
| Detail View | `Shift + left-click` | Run PAMG from the clicked seed and add the result to the current mask |
| Detail View | Left-click or left-button drag | Paint foreground |
| Detail View | Right-click or right-button drag | Erase foreground |
| Detail View | Mouse wheel | Adjust the brush size |
| Detail View | `Ctrl + mouse wheel` | Adjust the local image magnification |

### Keyboard shortcuts

| Shortcut | Function |
|---|---|
| `A` | Previous image; also stops playback if active |
| `D` | Next image; also stops playback if active |
| `Tab` | Cycle through the target boxes in the current image |
| `Space` | Play or pause the image sequence |
| `Delete` | Delete the selected target box |
| `Ctrl+S` | Save the current mask |
| `Ctrl+C` | Clear the current mask |
| `Ctrl+Z` | Undo the most recent mask operation |
| `Ctrl+Shift+Z` | Undo the most recent target-box operation |
| Hold `Ctrl` | Temporarily hide the mask overlay; release to show it again |

Mask and target-box histories each retain the 10 most recent undo states. Target
boxes are used only to define local editing regions. Creating, moving, resizing,
or deleting a target box does not update the YOLO `.txt` file.

### Parameters and visualization

- **Pad**: add context around target boxes loaded from YOLO annotations. Changing
  this value recalculates the boxes from the original YOLO annotations.
- **R-Scale**: magnification of the Detail View; it can also be adjusted with
  `Ctrl + mouse wheel`.
- **Rs**: PAMG growth-scale prior.
- **Mode**: target polarity. `1:Bright` selects bright targets, `-1:Dark`
  selects dark targets, and `0:Auto` determines the polarity automatically.
  Click **Apply Algo** after changing `Rs` or Mode.
- **CLAHE Enhance**: enhance display contrast without modifying the original
  image or saved mask.
- **Color**: switch among Gray, JET, and OCEAN display modes without affecting
  saved results.
- **Gamma / Alpha**: adjust the intensity and opacity of the displayed mask
  overlay without modifying mask values.
- **Reset Params**: restore the default PAMG, brush, and visualization settings.
- **Play / Stop / Spd**: play or stop the image sequence and set the interval
  between frames in milliseconds.
- **Clear Mask**: clear the current mask; this operation can be undone with
  `Ctrl+Z`.

## SIRSTD dataset

SIRSTD contains 47,250 infrared frames.

- Dataset archive: `SIRSTD-Pixel.zip`
- Download: [Baidu Netdisk](https://pan.baidu.com/s/1HCL8RU122I2xT_G7ey331w?pwd=3j72)
  (access code: `3j72`)

The dataset is not stored in this Git repository. It is distributed separately
under [CC BY-NC 4.0](DATASET_LICENSE.md).

Extract or copy the dataset into the tracked placeholder directories:

```text
dataset/SIRSTD/
  images/
    IR_2_00000.jpg
  masks/
    IR_2_00000.png
  YOLO_labels/
    IR_2_00000.txt
  splits/
    train.txt
    test.txt
```

`YOLO_labels` contains standard YOLO bounding-box annotations. A negative sample
may use an empty annotation file or have no corresponding annotation file. Each
split file contains one sample ID per line, such as `IR_2_00000`.

## Project layout

```text
./
  RPRNet/
    configs/          editable training hyperparameters
    datasets/         temporal frames and online mask supervision
    nets/             RPR-Net and its training loss
    tools/            training and unified sequence/dataset inference
    utils/            decoding, mask rendering, and checkpoint I/O
  dataset/SIRSTD/     local dataset placeholder (images, masks, YOLO labels, and splits)
  labelSIRST/         PAMG and Label-IRST
  weights/            official model parameters
  README.md
  DATASET_LICENSE.md
  LICENSE
  requirements.txt
```

## Citation

If you use the dataset or code from this repository in your research, you must
cite the following paper:

```bibtex
@ARTICLE{11683233,
  author={Gao, Weihua and Niu, Wenlong and Tang, Jie and Yang, Man and Zhang, Jiafeng and Peng, Xiaodong and Zhan, Yang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={Point-to-Mask: A Two-Stage Framework for Infrared Small Target Detection with Arbitrary Point Supervision},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2026.3730901}}
```

## License

The code and model weights are released under the custom
[Point2Mask Research-Only Non-Commercial License](LICENSE). Modification and
redistribution are allowed only for non-commercial research under the same
license and with preserved attribution. This is a research-use source release,
not an OSI-approved open-source license.

The SIRSTD dataset is licensed separately under
[CC BY-NC 4.0](DATASET_LICENSE.md). This dataset license covers the dataset only
and does not replace the code and model-weight license above.
