# ARD100 Dataset Generation Guide

This guide provides step-by-step instructions to generate the complete ARD100_mask32 dataset structure from raw video files.

## Prerequisites

Before starting, ensure you have:

- ARD100 raw video files (phantom*.mp4)

- Manual annotations in VOC XML format

- Python environment with OpenCV, NumPy installed

- Sufficient disk space (videos + extracted frames + masks)

## Directory Structure Overview

You will need to set up the following directory structure:

```
/path/to/your/workspace/
├── ARD-MAV/                          # Raw videos
│   └── test_videos/
│       ├── phantom02.mp4
│       ├── phantom03.mp4
│       └── ... (all 100 videos)
│
├── phantom-dataset/                   # Intermediate processing
│   ├── images/                        # Extracted frames
│   │   ├── phantom02/
│   │   ├── phantom03/
│   │   └── ...
│   ├── Annotations/                   # Manual VOC annotations
│   │   ├── phantom02/
│   │   ├── phantom03/
│   │   └── ...
│   └── mask32/                        # Generated motion masks
│       ├── phantom02/
│       ├── phantom03/
│       └── ...
│
└── YOLOMG/
    └── datasets/
        └── ARD100_mask32/             # Final organized dataset
            ├── images/                 # All RGB images (flattened)
            ├── Annotations/            # All XML annotations (flattened)
            ├── labels/                 # YOLO format labels
            ├── mask32/                 # All motion masks (flattened)
            ├── ImageSets/
            │   └── Main/
            │       ├── train.txt
            │       ├── val.txt
            │       └── test.txt
            ├── train.txt               # Full paths to training RGB images
            ├── train2.txt              # Full paths to training masks
            ├── val.txt                 # Full paths to validation RGB images
            ├── val2.txt                # Full paths to validation masks
            └── test.txt                # Full paths to test RGB images
```

## Step-by-Step Instructions

### Step 1: Extract Frames from Videos

**Purpose**: Convert video files into individual frame images.

**Script**: `test_code/YOLOMG_extract_frames.py`

**Instructions**:

1. Create the directory structure:

```bash
mkdir -p /path/to/phantom-dataset/images
```

1. Modify `YOLOMG_extract_frames.py` to set your paths:

```python
video_folder = '/path/to/ARD-MAV/test_videos/'
image_folder = '/path/to/phantom-dataset/images/'
```

1. Run the extraction:

```bash
cd /path/to/YOLOMG/test_code
python YOLOMG_extract_frames.py
```

**Output**: Each video will create a subdirectory with extracted frames:

- `/phantom-dataset/images/phantom02/phantom02_0001.jpg`

- `/phantom-dataset/images/phantom02/phantom02_0002.jpg`

- etc.

**Note**: This extracts **every frame** from the video. A typical video may produce hundreds to thousands of frames.

---

### Step 2: Manually Annotate Frames

**Purpose**: Create bounding box annotations for drones in each frame.

**Tools**: Use any annotation tool that supports VOC XML format (e.g., LabelImg, CVAT, VGG Image Annotator).

**Instructions**:

1. Create annotations directory:

```bash
mkdir -p /path/to/phantom-dataset/Annotations
```

1. For each video subdirectory in `images/`, create corresponding annotation files:

- Annotate drones with bounding boxes

- Save as VOC XML format

- Use class name: `"UAV"` or `"Drone"`

1. Organize annotations to match image structure:

```
/phantom-dataset/Annotations/
├── phantom02/
│   ├── phantom02_0001.xml
│   ├── phantom02_0002.xml
│   └── ...
├── phantom03/
└── ...
```

**XML Format Example**:

```xml
<annotation>
  <folder>phantom02</folder>
  <filename>phantom02_0001.jpg</filename>
  <size>
    <width>1920</width>
    <height>1080</height>
    <depth>3</depth>
  </size>
  <object>
    <name>UAV</name>
    <difficult>0</difficult>
    <bndbox>
      <xmin>850</xmin>
      <ymin>450</ymin>
      <xmax>920</xmax>
      <ymax>520</ymax>
    </bndbox>
  </object>
</annotation>
```

---

### Step 3: Generate Motion Masks (mask32)

**Purpose**: Create pixel-level motion masks using 5-frame differencing.

**Script**: `test_code/generate_mask5.py` and `test_code/FD5_mask.py`

**Instructions**:

1. Ensure you have the MOD_Functions.py module (contains motion_compensate function).

1. Modify `generate_mask5.py` to set your paths:

```python
# Line 38 in generate_mask5.py
cap = cv2.VideoCapture('/path/to/ARD-MAV/test_videos/' + video_name + '.mp4')
```

1. Modify `FD5_mask.py` to set output path:

```python
# Line 71 in FD5_mask.py
save_path = '/path/to/phantom-dataset/mask32/' + video_name
```

1. Choose which video set to process by modifying line 36 in `generate_mask5.py`:

```python
# For training videos
for video_sets in sets:  # Training set (65 videos)

# For test videos
for video_sets in set0:  # Test set (35 videos)
```

1. Run the mask generation:

```bash
cd /path/to/YOLOMG/test_code
python generate_mask5.py
```

**Output**: Motion masks for each frame (starting from frame 3, since 5 frames are needed):

- `/phantom-dataset/mask32/phantom02/phantom02_0001.jpg`

- `/phantom-dataset/mask32/phantom02/phantom02_0002.jpg`

- etc.

**Note**: The mask generation uses:

- Gaussian blur for noise reduction

- Motion compensation using homography

- Frame differencing between frames (t-2, t, t+2)

- Averaged difference for robust motion detection

---

### Step 4: Organize Dataset (Copy to Final Location)

**Purpose**: Copy images, masks, and annotations to the final dataset directory, filtering by annotation quality and object size.

**Script**: `test_code/generate_dataset.py`

**Instructions**:

1. Create the final dataset directory:

```bash
mkdir -p /path/to/YOLOMG/datasets/ARD100_mask32/{images,Annotations,mask32}
```

1. Modify `generate_dataset.py` paths (lines 64-70):

```python
# Source directories
imgdir = "/path/to/phantom-dataset/images/" + id + "/"
annodir = '/path/to/phantom-dataset/Annotations/' + id + '/'
maskdir = '/path/to/phantom-dataset/mask32/' + id + '/'

# Destination directories
imgdest = '/path/to/YOLOMG/datasets/ARD100_mask32/images/'
annodest = '/path/to/YOLOMG/datasets/ARD100_mask32/Annotations/'
maskdest = '/path/to/YOLOMG/datasets/ARD100_mask32/mask32/'
```

1. Choose which dataset split to process (line 62):

```python
# For training videos (65 videos)
for video_sets in set1:

# For test videos (35 videos)
for video_sets in set2:
```

1. Run the organization script:

```bash
cd /path/to/YOLOMG/test_code
python generate_dataset.py
```

**What it does**:

- Iterates through each video's frames

- Checks if annotation exists

- Filters out objects smaller than 25 pixels² (too small to detect reliably)

- Copies valid image, mask, and annotation triplets to destination

- Flattens directory structure (all files in single directories)

**Output**: Flattened dataset structure:

```
ARD100_mask32/
├── images/
│   ├── phantom02_0002.jpg
│   ├── phantom02_0003.jpg
│   ├── phantom03_0002.jpg
│   └── ...
├── Annotations/
│   ├── phantom02_0002.xml
│   ├── phantom02_0003.xml
│   └── ...
└── mask32/
    ├── phantom02_0002.jpg
    ├── phantom02_0003.jpg
    └── ...
```

---

### Step 5: Split Dataset into Train/Val/Test

**Purpose**: Randomly split the dataset into training, validation, and test sets.

**Script**: `data/split_train_val.py`

**Instructions**:

1. Create ImageSets directory:

```bash
mkdir -p /path/to/YOLOMG/datasets/ARD100_mask32/ImageSets/Main
```

1. Run the split script:

```bash
cd /path/to/YOLOMG/data
python split_train_val.py \
  --xml_path /path/to/YOLOMG/datasets/ARD100_mask32/Annotations \
  --txt_path /path/to/YOLOMG/datasets/ARD100_mask32/ImageSets/Main
```

**Parameters**:

- `trainval_percent = 1.0` (100% used for train+val, 0% for separate test)

- `train_percent = 0.85` (85% of trainval for training, 15% for validation)

**Output**: Text files with image IDs (without extensions):

```
ImageSets/Main/
├── train.txt       # 85% of data
├── val.txt         # 15% of data
├── trainval.txt    # 100% (train + val)
└── test.txt        # Empty (or separate test set)
```

**Note**: For ARD100, you should modify the script to use the predefined train/test split based on video IDs rather than random splitting. The dataset has specific training videos (set1) and test videos (set2).

---

### Step 6: Convert VOC Annotations to YOLO Format

**Purpose**: Convert XML annotations to YOLO format (normalized center x, y, width, height).

**Script**: `data/voc2yolo.py`

**Instructions**:

1. Create labels directory:

```bash
mkdir -p /path/to/YOLOMG/datasets/ARD100_mask32/labels
```

1. Modify `voc2yolo.py` paths (lines 26-27, 60, 62):

```python
# Line 26-27
in_file = open('/path/to/YOLOMG/datasets/ARD100_mask32/Annotations/%s.xml' % (image_id), encoding='UTF-8')
out_file = open('/path/to/YOLOMG/datasets/ARD100_mask32/labels/%s.txt' % (image_id), 'w')

# Line 60
if not os.path.exists('/path/to/YOLOMG/datasets/ARD100_mask32/labels/'):
    os.makedirs('/path/to/YOLOMG/datasets/ARD100_mask32/labels/')

# Line 62
image_ids = open('/path/to/YOLOMG/datasets/ARD100_mask32/ImageSets/Main/%s.txt' % image_set).read().strip().split()
```

1. Update the class name (line 6) to match your annotations:

```python
classes = ["UAV"]  # or ["Drone"] depending on your annotation class name
```

1. Run the conversion:

```bash
cd /path/to/YOLOMG/data
python voc2yolo.py
```

**Output**: YOLO format labels:

```
labels/
├── phantom02_0002.txt
├── phantom02_0003.txt
└── ...
```

**YOLO Label Format**: `class_id center_x center_y width height` (all normalized 0-1)

```
0 0.4531 0.5208 0.0365 0.0648
```

---

### Step 7: Generate Image Path Files (train.txt, val.txt, test.txt)

**Purpose**: Create text files with absolute paths to RGB images for each dataset split.

**Script**: `data/voc_label.py`

**Instructions**:

1. Modify `voc_label.py` paths (lines 13-14, 16):

```python
# Line 13
image_ids = open('/path/to/YOLOMG/datasets/ARD100_mask32/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()

# Line 14
list_file = open('/path/to/YOLOMG/datasets/ARD100_mask32/%s.txt' % (image_set), 'w')

# Line 16
list_file.write('/path/to/YOLOMG/datasets/ARD100_mask32/images/%s.jpg\n' % (image_id))
```

1. Run the script:

```bash
cd /path/to/YOLOMG/data
python voc_label.py
```

**Output**: Path files for RGB images:

```
ARD100_mask32/
├── train.txt
├── val.txt
└── test.txt
```

**Content Example** (train.txt):

```
/path/to/YOLOMG/datasets/ARD100_mask32/images/phantom09_0002.jpg
/path/to/YOLOMG/datasets/ARD100_mask32/images/phantom09_0003.jpg
/path/to/YOLOMG/datasets/ARD100_mask32/images/phantom10_0002.jpg
...
```

---

### Step 8: Generate Mask Path Files (train2.txt, val2.txt)

**Purpose**: Create text files with absolute paths to motion mask images for each dataset split.

**Script**: `data/voc_label2.py`

**Instructions**:

1. Modify `voc_label2.py` paths (lines 13-14, 16):

```python
# Line 13
image_ids = open('/path/to/YOLOMG/datasets/ARD100_mask32/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()

# Line 14
list_file = open('/path/to/YOLOMG/datasets/ARD100_mask32/%s2.txt' % (image_set), 'w')

# Line 16
list_file.write('/path/to/YOLOMG/datasets/ARD100_mask32/mask32/%s.jpg\n' % (image_id))
```

1. Run the script:

```bash
cd /path/to/YOLOMG/data
python voc_label2.py
```

**Output**: Path files for motion masks:

```
ARD100_mask32/
├── train2.txt
├── val2.txt
└── test2.txt (if applicable)
```

**Content Example** (train2.txt):

```
/path/to/YOLOMG/datasets/ARD100_mask32/mask32/phantom09_0002.jpg
/path/to/YOLOMG/datasets/ARD100_mask32/mask32/phantom09_0003.jpg
/path/to/YOLOMG/datasets/ARD100_mask32/mask32/phantom10_0002.jpg
...
```

---

### Step 9: Update Dataset Configuration YAML

**Purpose**: Configure the dataset paths for training.

**File**: `data/ARD100_mask32.yaml`

**Instructions**:

1. Edit `data/ARD100_mask32.yaml`:

```yaml
# Train/val/test sets
train: /path/to/YOLOMG/datasets/ARD100_mask32/train.txt
train2: /path/to/YOLOMG/datasets/ARD100_mask32/train2.txt
val: /path/to/YOLOMG/datasets/ARD100_mask32/val.txt
val2: /path/to/YOLOMG/datasets/ARD100_mask32/val2.txt
test: /path/to/YOLOMG/datasets/ARD100_mask32/test.txt

# Classes
nc: 1  # number of classes
names: ['Drone']  # class names
```

1. Ensure the paths are absolute and match your system.

---

## Final Dataset Structure

After completing all steps, your final dataset should look like:

```
/path/to/YOLOMG/datasets/ARD100_mask32/
├── images/                           # ~10,000+ RGB frame images
│   ├── phantom09_0002.jpg
│   ├── phantom09_0003.jpg
│   └── ...
├── Annotations/                      # ~10,000+ VOC XML annotations
│   ├── phantom09_0002.xml
│   ├── phantom09_0003.xml
│   └── ...
├── labels/                           # ~10,000+ YOLO format labels
│   ├── phantom09_0002.txt
│   ├── phantom09_0003.txt
│   └── ...
├── mask32/                           # ~10,000+ motion mask images
│   ├── phantom09_0002.jpg
│   ├── phantom09_0003.jpg
│   └── ...
├── ImageSets/
│   └── Main/
│       ├── train.txt                 # Image IDs for training
│       ├── val.txt                   # Image IDs for validation
│       └── test.txt                  # Image IDs for testing
├── train.txt                         # Full paths to training RGB images
├── train2.txt                        # Full paths to training masks
├── val.txt                           # Full paths to validation RGB images
├── val2.txt                          # Full paths to validation masks
└── test.txt                          # Full paths to test RGB images
```

---

## Verification Checklist

Before training, verify:

- [ ] All images have corresponding masks with the same filename

- [ ] All images have corresponding annotations (XML and TXT)

- [ ] Path files (train.txt, train2.txt, etc.) contain absolute paths

- [ ] Paths in YAML configuration file are correct

- [ ] Number of lines in train.txt matches train2.txt

- [ ] Number of lines in val.txt matches val2.txt

- [ ] Class names in YAML match annotation class names

- [ ] All paths are accessible and files exist

**Quick verification command**:

```bash
# Check if all images have masks
cd /path/to/YOLOMG/datasets/ARD100_mask32
ls images/ | wc -l
ls mask32/ | wc -l
ls labels/ | wc -l

# Check path file line counts
wc -l train.txt train2.txt val.txt val2.txt
```

---

## Training Command

Once the dataset is ready, you can start training:

```bash
cd /path/to/YOLOMG

# Single GPU training
python train.py \
  --data data/ARD100_mask32.yaml \
  --cfg models/dual_uav2.yaml \
  --weights yolov5s.pt \
  --batch-size 8 \
  --epochs 100 \
  --imgsz 1280 \
  --name ARD100_mask32-1280_uavs \
  --device 0

# Multi-GPU DDP training
python -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port 12345 \
  train.py \
  --data data/ARD100_mask32.yaml \
  --cfg models/dual_uav2.yaml \
  --weights yolov5s.pt \
  --batch-size 16 \
  --epochs 100 \
  --imgsz 1280 \
  --name ARD100_mask32-1280_uavs \
  --device 0,1,2,3
```

---

## Troubleshooting

### Issue: "File not found" errors during training

- **Solution**: Verify all paths in YAML and path files are absolute paths

- Check file permissions

### Issue: Mismatched number of images and masks

- **Solution**: Ensure `generate_dataset.py` copied matching triplets

- Re-run Step 4 if needed

### Issue: Motion masks are all black

- **Solution**: Check video quality and motion compensation parameters

- Verify `MOD_Functions.py` is available and working

- Try different frame differencing methods (FD2, FD3)

### Issue: Training fails with "no objects found"

- **Solution**: Verify YOLO labels are correctly formatted

- Check class names match between annotations and YAML

- Ensure bounding boxes are within image bounds

### Issue: Out of memory during mask generation

- **Solution**: Process videos in smaller batches

- Reduce video resolution if possible

- Close other applications

---

## Summary

The complete workflow is:

1. **Extract frames** from videos → `images/`

1. **Manually annotate** frames → `Annotations/`

1. **Generate motion masks** → `mask32/`

1. **Organize dataset** → Copy to final location

1. **Split dataset** → `train.txt`, `val.txt`, `test.txt` (IDs only)

1. **Convert labels** → VOC XML to YOLO TXT

1. **Generate image paths** → `train.txt`, `val.txt`, `test.txt` (full paths)

1. **Generate mask paths** → `train2.txt`, `val2.txt`

1. **Update YAML** → Configure dataset paths

1. **Train model** → Run training script

This process transforms raw videos into a fully prepared dual-input dataset ready for YOLOMG training.

