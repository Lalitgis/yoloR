# yolor <img src="man/figures/logo.png" align="right" height="80"/>

**R-Native YOLO Object Detection — Annotate, Train, Detect, Evaluate**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![R](https://img.shields.io/badge/R-%3E%3D4.1-blue.svg)](https://cran.r-project.org/)
[![GitHub](https://img.shields.io/badge/GitHub-Lalitgis%2FyoloR-black?logo=github)](https://github.com/Lalitgis/yoloR)

---

## Overview

`yolor` is a complete R package for **YOLO-based object detection and instance segmentation** — from raw annotated images to trained model to evaluated results — without ever leaving R.

It bridges the gap between popular annotation tools and the Ultralytics YOLOv8 training engine by providing clean R functions for every step of the workflow:

```
Annotate                   Prepare                  Train & Detect            Evaluate
────────                   ───────                  ──────────────            ────────
ShinyLabel   ──┐                                    yolo_train()     ──►      yolo_metrics()
Roboflow     ──┼──► import ──► sl_export_dataset() ──► data.yaml              yolo_benchmark()
LabelImg     ──┤             sl_class_summary()    yolo_detect()    ──►      metrics_export()
CVAT         ──┘             yolo_validate_dataset() plot()                   metrics_compare()
COCO / VOC ──┘
```

Whether you have images annotated in **ShinyLabel**, **Roboflow**, **LabelImg**, **CVAT**, **Label Studio**, or any other tool — `yolor` reads them all and gets you to a trained model in minutes.

---

## Installation

```r
# Install from GitHub
devtools::install_github("Lalitgis/yoloR")

# One-time Python backend setup (Ultralytics YOLOv8)
library(yolor)
yolo_setup()   # creates a 'yolor' Python virtualenv with ultralytics + torch

# Add this line to every script
reticulate::use_virtualenv("yolor")
```

**System requirements:**
- R ≥ 4.1
- Python ≥ 3.8 (installed separately from [python.org](https://www.python.org/downloads/))
- Windows / macOS / Linux

---

## Quick Start

### From ShinyLabel

```r
library(yolor)
reticulate::use_virtualenv("yolor")

# Read annotations from ShinyLabel database
ds <- sl_read_db("project.db")
print(ds)           # summary: images, classes, box counts
plot(ds)            # class distribution bar chart
sl_class_summary(ds) # per-class annotation counts

# Export to YOLO dataset
sl_export_dataset(ds, output_dir = "dataset/", val_split = 0.2)

# Train
model  <- yolo_model("yolov8n")
result <- yolo_train(model, data = "dataset/data.yaml", epochs = 100)

# Detect
preds <- yolo_detect(result, images = "new_images/", conf = 0.4)
plot(preds)

# Metrics
m <- yolo_metrics(result, data = "dataset/data.yaml")
plot(m, type = "dashboard")
metrics_export(m, dir = "results/")
```

### From Roboflow

```r
# Export as YOLOv8 PyTorch in Roboflow → unzip → point here
yaml <- rf_load_yolo("roboflow_export/")
result <- yolo_train(yolo_model("yolov8n"), data = yaml, epochs = 50)
```

### From an existing YOLO dataset

```r
# If you already have images/ + labels/ + data.yaml
setwd("yolo_dataset/")
yolo_validate_dataset(".")
result <- yolo_train(yolo_model("yolov8n"), data = "data.yaml", epochs = 50)
```

### From LabelImg / CVAT / Label Studio

```r
# Auto-detect format and prepare for training in one call
yaml <- import_annotations(
  path       = "my_annotated_folder/",
  output_dir = "dataset/"
)
result <- yolo_train(yolo_model("yolov8n"), data = yaml, epochs = 50)
```

---

## Annotation Sources Supported

| Tool / Format | Function | Notes |
|---|---|---|
| **ShinyLabel** SQLite `.db` | `sl_read_db()` | Native format, full metadata |
| **ShinyLabel** CSV export | `sl_read_csv()` | Standard CSV with bbox columns |
| **Roboflow** YOLOv8 export | `rf_load_yolo()` | Zero conversion — use directly |
| **Roboflow** COCO JSON | `rf_coco_to_yolo()` | Auto-converts to YOLO layout |
| **Roboflow** CSV export | `rf_read_csv()` | Reads Roboflow CSV columns |
| **Roboflow** API download | `rf_download()` | Direct API with your key |
| **LabelImg YOLO** `.txt` | `import_yolo_labels()` | Reads `.txt` + `classes.txt` |
| **LabelImg VOC** `.xml` | `import_voc_xml()` | Pascal VOC bounding boxes |
| **CVAT / Label Studio** COCO | `import_coco_json()` | COCO JSON format |
| **Any folder** | `import_annotations()` | Auto-detects format |
| **Existing YOLO dataset** | `yolo_validate_dataset()` | Use data.yaml directly |

---

## Key Functions

### Annotation Import

| Function | Description |
|---|---|
| `sl_read_db(path)` | Read ShinyLabel SQLite database |
| `sl_read_csv(path)` | Read ShinyLabel CSV export |
| `sl_export_dataset(ds, dir)` | Export to YOLO folder layout with train/val split |
| `sl_class_summary(ds)` | Annotation counts per class |
| `import_annotations(path)` | Auto-detect format and import |
| `import_yolo_labels(images, labels)` | Import YOLO `.txt` label files |
| `import_voc_xml(dir)` | Import Pascal VOC XML files |
| `import_coco_json(json)` | Import COCO JSON annotations |

### Roboflow

| Function | Description |
|---|---|
| `rf_load_yolo(dir)` | Load Roboflow YOLOv8 export directly |
| `rf_coco_to_yolo(dir)` | Convert Roboflow COCO JSON to YOLO |
| `rf_read_csv(path)` | Read Roboflow CSV export |
| `rf_download(workspace, project, version)` | Download from Roboflow API |
| `rf_summary(dir)` | Dataset overview (images / labels / boxes per split) |

### Models

| Function | Description |
|---|---|
| `yolo_setup()` | Install Ultralytics Python backend |
| `yolo_model(weights)` | Load YOLOv8 model (auto-downloads pretrained weights) |
| `yolo_available_models()` | List all available model sizes |

### Training

| Function | Description |
|---|---|
| `yolo_train(model, data)` | Fine-tune on your dataset |
| `yolo_resume(run_dir)` | Resume interrupted training |
| `yolo_validate_dataset(dir)` | Sanity-check dataset before training |

### Detection & Segmentation

| Function | Description |
|---|---|
| `yolo_detect(model, images)` | Run inference on images / directory / URL |
| `plot(results)` | Visualise bounding boxes with ggplot2 |
| `as_tibble(results)` | Flatten detections to a data frame |
| `yolo_export_csv(results, path)` | Save detections to CSV |
| `yolo_export_geojson(results, path)` | Save detections to GeoJSON |

### Accuracy Metrics

| Function | Description |
|---|---|
| `yolo_benchmark(model, data)` | Quick mAP / Precision / Recall / F1 |
| `yolo_metrics(model, data)` | Full metrics: PR curve, F1 curve, confusion matrix |
| `metrics_from_predictions(pred, gt)` | Pure-R metrics — no Python required |
| `plot(metrics, type)` | 6 plot types: PR, F1, confusion, bar, radar, dashboard |
| `metrics_export(metrics, dir)` | Export CSV / JSON / PNG / HTML report |
| `metrics_compare(m1, m2)` | Side-by-side model comparison chart |

---

## Supported YOLO Models

| Model | Parameters | Speed | Best For |
|---|---|---|---|
| `yolov8n` | 3.2M | ⚡⚡⚡ Fastest | Edge devices, real-time, small datasets |
| `yolov8s` | 11.2M | ⚡⚡ Fast | Balanced speed and accuracy |
| `yolov8m` | 25.9M | ⚡ Medium | General purpose |
| `yolov8l` | 43.7M | 🐢 Slow | High accuracy, server GPU |
| `yolov8x` | 68.2M | 🐢🐢 Slowest | Maximum accuracy |
| `yolov8n-seg` | 3.4M | ⚡⚡⚡ Fastest | Instance segmentation (nano) |
| `yolov8s-seg` | 11.8M | ⚡⚡ Fast | Instance segmentation (small) |

---

## Accuracy Metrics — 6 Visualisation Types

```r
m <- yolo_metrics(model, data = "dataset/data.yaml")

plot(m, type = "pr")          # Precision-Recall curve
plot(m, type = "f1")          # F1 vs Confidence threshold
plot(m, type = "confusion")   # Confusion matrix heatmap
plot(m, type = "bar")         # Per-class bar chart
plot(m, type = "radar")       # Overall metric spider chart
plot(m, type = "dashboard")   # All plots in one 2x2 panel
```

Export everything in one call:
```r
metrics_export(m, dir = "results/")
# Saves: overall_metrics.csv, per_class_metrics.csv,
#        pr_curve.csv, f1_curve.csv, confusion_matrix.csv,
#        metrics.json, 5 PNGs, metrics_report.html
```

---

## Example Scripts

Three runnable examples are bundled with the package:

```r
# See all examples
source(system.file("examples", "00_index.R", package = "yolor"))

# Open in RStudio
file.edit(system.file("examples", "01_object_detection.R", package = "yolor"))
file.edit(system.file("examples", "02_segmentation.R",     package = "yolor"))
file.edit(system.file("examples", "03_metrics.R",          package = "yolor"))
```

| Script | Covers |
|---|---|
| `01_object_detection.R` | Full detection pipeline: annotate → train → detect → export |
| `02_segmentation.R` | YOLOv8-seg: train → masks → metrics → export |
| `03_metrics.R` | All metric types, visualisations, export, model comparison |

---

## Built-in Help

```r
??yolor                  # full function index
?yolo_train              # training help with examples
?yolo_metrics            # metrics help
?import_annotations      # annotation import help
?yolor_examples          # examples guide
```

---

## Architecture

```
yolor/
├── R/
│   ├── sl_read.R              # ShinyLabel DB + CSV reader & exporter
│   ├── roboflow.R             # Roboflow adapter (YOLOv8 / COCO / CSV / API)
│   ├── import_annotations.R   # Import YOLO TXT / VOC XML / COCO JSON
│   ├── yolo_model.R           # Model loading, setup, backend management
│   ├── yolo_train.R           # Training wrapper + result S3 class
│   ├── yolo_detect.R          # Inference + ggplot2 visualisation
│   ├── yolo_benchmark.R       # Quick evaluation (mAP / P / R / F1)
│   ├── yolo_metrics.R         # Full metrics computation engine
│   ├── yolo_metrics_plot.R    # PR / F1 / confusion / radar / dashboard
│   ├── yolo_metrics_export.R  # CSV / JSON / PNG / HTML export
│   ├── utils.R                # GeoJSON, draw boxes, dataset validation
│   └── examples.R             # Bundled example data helpers
├── man/                       # Help pages for all functions
├── inst/
│   ├── extdata/
│   │   └── example_annotations.db   # Bundled example ShinyLabel DB
│   └── examples/
│       ├── 01_object_detection.R
│       ├── 02_segmentation.R
│       └── 03_metrics.R
└── tests/testthat/            # 100+ unit tests, all pure R
```

---

## Common Workflows

**I have a ShinyLabel project:**
```r
ds <- sl_read_db("project.db")
sl_export_dataset(ds, "dataset/")
result <- yolo_train(yolo_model("yolov8n"), data = "dataset/data.yaml")
```

**I have a Roboflow project:**
```r
yaml <- rf_load_yolo("roboflow_yolov8_export/")  # or rf_download(...)
result <- yolo_train(yolo_model("yolov8n"), data = yaml)
```

**I have images annotated with LabelImg:**
```r
yaml <- import_annotations("my_folder/", output_dir = "dataset/")
result <- yolo_train(yolo_model("yolov8n"), data = yaml)
```

**I already have a YOLO dataset:**
```r
setwd("yolo_dataset/")
result <- yolo_train(yolo_model("yolov8n"), data = "data.yaml", epochs = 50)
```

**I want to evaluate without Python:**
```r
m <- metrics_from_predictions(predictions_tibble, ground_truth_tibble)
plot(m, type = "dashboard")
metrics_export(m, dir = "results/")
```

---

## License

MIT © Lalit GIS  
Built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [ShinyLabel](https://github.com/Lalitgis/ShinyLabel)
