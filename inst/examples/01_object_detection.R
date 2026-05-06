# =============================================================
#  Example 04: Full Training + Metrics — Real Dataset Workflow
#  yolor package
#
#  This example covers a complete real-world workflow using an
#  existing YOLO dataset (images/ + labels/ + data.yaml/.yml).
#  All known issues are handled and fixed:
#
#  Fixes applied:
#   ✔ plot.yolo_metrics() called explicitly (avoids base R conflict)
#   ✔ plot.yolo_benchmark() called explicitly
#   ✔ plot.yolo_results() called explicitly
#   ✔ yolor::as_tibble.yolo_results() used (avoids dplyr conflict)
#   ✔ PR curve built from raw predictions when Ultralytics skips it
#   ✔ Low confidence threshold for detection (0.05 not 0.4)
#   ✔ Ground truth built from label .txt files for pure-R metrics
#   ✔ Both .yaml and .yml accepted automatically
#
#  Requirements:
#   - An existing YOLO dataset folder with:
#       images/train/  images/val/
#       labels/train/  labels/val/
#       data.yaml  (or data.yml)
#   - yolor installed and Python backend configured
#
#  Run once per machine to set up:
#   yolo_setup()
# =============================================================

library(yolor)
library(dplyr)     # needed for as_tibble, bind_rows etc.
library(magick)    # needed for reading image dimensions

# Activate Python virtualenv
reticulate::use_virtualenv("yolor")


# ── 0. Configure your paths ───────────────────────────────────
# Change DATASET_DIR to your actual folder path

DATASET_DIR <- "C:/Users/Hp/Downloads/yolo_annotations_20260503/yolo_dataset"

# Run name for this training session — change for new runs
RUN_NAME  <- "train_v1"
RUN_DIR   <- file.path(DATASET_DIR, "runs", "detect", "runs", RUN_NAME)

setwd(DATASET_DIR)
cat("Working directory:", getwd(), "\n")


# ── 1. Auto-detect YAML file (.yaml or .yml) ─────────────────

YAML_FILE <- if (file.exists("data.yaml")) "data.yaml" else "data.yml"
cat("Using:", YAML_FILE, "\n\n")


# ── 2. Validate dataset ───────────────────────────────────────

cat("========== Dataset Validation ==========\n")
yolo_validate_dataset(".")
rf_summary(".")          # images / labels / boxes per split


# ── 3. Choose model and train ─────────────────────────────────
#
#  Model sizes:
#   yolov8n  —  3.2M params  fastest   small datasets / CPU
#   yolov8s  — 11.2M params  fast      balanced
#   yolov8m  — 25.9M params  medium    general purpose
#   yolov8l  — 43.7M params  slow      high accuracy
#   yolov8x  — 68.2M params  slowest   maximum accuracy
#
#  Tips for small datasets (< 100 images):
#   - Use yolov8n — it avoids overfitting
#   - Increase epochs to 100-200
#   - Use patience = 30 (early stopping)
#   - Add more images if Precision stays near 0

cat("\n========== Loading Model ==========\n")

model <- yolo_model(
  weights = "yolov8n",
  task    = "detect",
  device  = "cpu"       # "cuda" for NVIDIA GPU, "mps" for Apple Silicon
)

cat("\n========== Training ==========\n")

result <- yolo_train(
  model,
  data      = YAML_FILE,
  epochs    = 50,        # increase to 100-200 for real projects
  imgsz     = 640,
  batch     = 8,         # reduce to 4 if memory errors occur
  lr0       = 0.01,
  patience  = 20,
  project   = "runs",
  name      = RUN_NAME,
  augment   = TRUE       # helps with small datasets
)

# Quick summary
cat("\n── Training Results ──\n")
cat(sprintf("  mAP@0.5      : %.4f\n", result$metrics$mAP50))
cat(sprintf("  mAP@0.5:0.95 : %.4f\n", result$metrics$mAP50_95))
cat(sprintf("  Precision    : %.4f\n", result$metrics$precision))
cat(sprintf("  Recall       : %.4f\n", result$metrics$recall))
cat("\nBest weights:", result$best_weights, "\n")


# ── 4. Load best model ────────────────────────────────────────

cat("\n========== Loading Best Trained Model ==========\n")

best_model <- yolo_model(
  result$best_weights,
  task   = "detect",
  device = "cpu"
)


# ── 5. Quick benchmark ────────────────────────────────────────

cat("\n========== Quick Benchmark ==========\n")

bench <- yolo_benchmark(
  best_model,
  data  = YAML_FILE,
  split = "val",
  conf  = 0.001,   # low conf = standard for mAP evaluation
  iou   = 0.6
)

cat("\n── Benchmark Results ──\n")
cat(sprintf("  mAP@0.5      : %.4f\n", bench$overall$mAP50))
cat(sprintf("  mAP@0.5:0.95 : %.4f\n", bench$overall$mAP50_95))
cat(sprintf("  Precision    : %.4f\n", bench$overall$precision))
cat(sprintf("  Recall       : %.4f\n", bench$overall$recall))
cat(sprintf("  F1           : %.4f\n", bench$overall$f1))

cat("\n── Per-class ──\n")
print(bench$per_class)

# FIX: call plot.yolo_benchmark() explicitly — avoids base R plot() conflict
plot.yolo_benchmark(bench)


# ── 6. Full metrics via Ultralytics ──────────────────────────

cat("\n========== Full Accuracy Metrics (Ultralytics) ==========\n")

metrics <- yolo_metrics(
  best_model,
  data  = YAML_FILE,
  split = "val",
  conf  = 0.001,
  iou   = 0.6
)

cat("\n── Overall ──\n")
print(metrics$overall)

cat("\n── Per-class ──\n")
print(metrics$per_class)

cat("\n── Confusion Matrix ──\n")
print(metrics$conf_matrix)


# ── 7. Visualise metrics ──────────────────────────────────────
#
#  FIX: Always call plot.yolo_metrics() explicitly.
#  Using plot(metrics, ...) triggers base R's plot() which errors
#  with: 'x' is a list, but does not have components 'x' and 'y'

cat("\n========== Visualisations ==========\n")

# F1-Confidence curve
# Shows optimal confidence threshold — dashed line = best threshold
cat("Plotting F1 curve...\n")
f1_plot <- plot.yolo_metrics(metrics, type = "f1")
print(f1_plot)

# Confusion matrix heatmap (recall-normalised)
# Diagonal = correct detections. Off-diagonal = misclassifications.
cat("Plotting confusion matrix...\n")
cm_plot <- plot.yolo_metrics(metrics, type = "confusion")
print(cm_plot)

# Per-class bar chart: Precision / Recall / F1 / AP per class
cat("Plotting per-class bar chart...\n")
bar_plot <- plot.yolo_metrics(metrics, type = "bar")
print(bar_plot)

# Radar / spider chart of overall metrics
cat("Plotting radar chart...\n")
radar_plot <- plot.yolo_metrics(metrics, type = "radar")
print(radar_plot)

# Dashboard — all plots in one 2x2 panel
# Requires patchwork: install.packages("patchwork")
if (requireNamespace("patchwork", quietly = TRUE)) {
  cat("Plotting dashboard...\n")
  dashboard <- plot.yolo_metrics(metrics, type = "dashboard")
  print(dashboard)
} else {
  cat("Install patchwork for the dashboard: install.packages('patchwork')\n")
}


# ── 8. PR curve via pure-R metrics ────────────────────────────
#
#  When the PR curve is empty (pr_curve tibble has 0 rows) it means
#  Ultralytics skipped it — usually because the val set is too small.
#  Fix: run detection at very low confidence and compute PR in pure R.

cat("\n========== PR Curve via Pure-R Metrics ==========\n")

if (nrow(metrics$pr_curve) == 0) {
  cat("PR curve empty from Ultralytics — computing from raw predictions...\n\n")

  # Detect on val images at very low confidence to capture all predictions
  raw_preds <- yolo_detect(
    best_model,
    images = "images/val/",
    conf   = 0.01,     # very low — captures everything the model sees
    iou    = 0.45,
    verbose = FALSE
  )

  # FIX: use yolor:: prefix for as_tibble to avoid dplyr conflict
  det_df <- yolor::as_tibble.yolo_results(raw_preds)
  cat("Raw predictions:\n")
  print(det_df)

  # Build ground truth tibble from val label .txt files
  # Each YOLO label line: class_id x_center y_center width height (normalised)
  val_labels <- list.files("labels/val", pattern = "\\.txt$",
                            full.names = TRUE)

  # Read class names from data.yaml
  cfg         <- yaml::read_yaml(YAML_FILE)
  class_names <- unlist(cfg$names)

  gt_list <- lapply(val_labels, function(lf) {
    stem     <- tools::file_path_sans_ext(basename(lf))
    img_path <- list.files("images/val",
                            pattern      = paste0("^", stem, "\\."),
                            full.names   = TRUE,
                            ignore.case  = TRUE)[1]

    lines <- readLines(lf, warn = FALSE)
    lines <- trimws(lines[nzchar(trimws(lines))])
    if (length(lines) == 0 || is.na(img_path)) return(NULL)

    # Read image dimensions to convert normalised → pixel coords
    info <- magick::image_info(magick::image_read(img_path))
    w <- info$width
    h <- info$height

    rows <- lapply(lines, function(l) {
      p      <- as.numeric(strsplit(l, "\\s+")[[1]])
      cls_id <- as.integer(p[1])
      xc <- p[2]; yc <- p[3]; bw <- p[4]; bh <- p[5]
      data.frame(
        image      = img_path,
        class_name = if (cls_id < length(class_names))
                       class_names[cls_id + 1L] else as.character(cls_id),
        xmin = (xc - bw / 2) * w,
        ymin = (yc - bh / 2) * h,
        xmax = (xc + bw / 2) * w,
        ymax = (yc + bh / 2) * h,
        stringsAsFactors = FALSE
      )
    })
    dplyr::bind_rows(rows)
  })

  gt <- dplyr::bind_rows(gt_list)

  cat("\nGround truth boxes:\n")
  print(gt)

  # Compute full metrics including PR curve — pure R, no Python needed
  m2 <- metrics_from_predictions(
    predictions  = det_df,
    ground_truth = gt,
    iou_thresh   = 0.5
  )

  cat("\n── Pure-R Metrics ──\n")
  print(m2)

  # Now plot PR curve — it will have real data
  cat("Plotting PR curve (pure-R)...\n")
  pr_plot <- plot.yolo_metrics(m2, type = "pr")
  print(pr_plot)

  # Full dashboard from pure-R metrics
  if (requireNamespace("patchwork", quietly = TRUE)) {
    cat("Plotting full dashboard (pure-R)...\n")
    dash2 <- plot.yolo_metrics(m2, type = "dashboard")
    print(dash2)
  }

} else {
  # PR curve data available from Ultralytics — plot directly
  pr_plot <- plot.yolo_metrics(metrics, type = "pr")
  print(pr_plot)
}


# ── 9. Run detection on val images ────────────────────────────
#
#  FIX: Use conf = 0.05 not 0.4 for small / early-trained models.
#  A model trained on very few images learns features but assigns
#  lower confidence scores — dropping to 0.05 reveals detections.

cat("\n========== Detection on Validation Images ==========\n")

val_img_dir <- file.path(DATASET_DIR, "images", "val")

preds <- yolo_detect(
  best_model,
  images  = val_img_dir,
  conf    = 0.05,    # FIX: low threshold for small/early models
                     # increase to 0.25-0.5 once you have more data
  iou     = 0.45,
  imgsz   = 640,
  verbose = TRUE
)

# FIX: use yolor:: prefix to avoid dplyr::as_tibble conflict
det_tibble <- yolor::as_tibble.yolo_results(preds)

cat("\n── Detection results ──\n")
print(preds)

cat("\n── Flat detections tibble ──\n")
print(det_tibble)

# Visualise detections on the first image
# FIX: call plot.yolo_results() explicitly — not plot()
if (preds$n_total > 0) {
  cat("Plotting detections...\n")
  det_plot <- plot.yolo_results(preds)
  print(det_plot)
} else {
  cat("No detections above conf=0.05.\n")
  cat("This means the model needs more training images.\n")
  cat("Aim for 100+ images per class for reliable detection.\n")
}


# ── 10. Export everything ─────────────────────────────────────

cat("\n========== Exporting Results ==========\n")

output_dir <- file.path(DATASET_DIR, "metrics_output")

# Choose which metrics object to export
# If PR curve was computed in step 8 use m2, otherwise use metrics
metrics_to_export <- if (exists("m2")) m2 else metrics

exported <- metrics_export(
  metrics_to_export,
  dir    = output_dir,
  plots  = c("pr", "f1", "confusion", "bar", "radar"),
  width  = 8,
  height = 6,
  dpi    = 150,
  html   = TRUE,
  pdf    = FALSE   # set TRUE if rmarkdown is installed
)

# Export detections if we have any
if (preds$n_total > 0) {
  yolo_export_csv(preds, file.path(output_dir, "detections.csv"))
  yolo_export_geojson(preds, file.path(output_dir, "detections.geojson"))
}

cat("\n── Exported files ──\n")
for (nm in names(exported)) {
  cat(sprintf("  %-22s : %s\n", nm, basename(exported[[nm]])))
}

# Open HTML report in browser
browseURL(exported$html_report)

cat("\n========== Done ==========\n")
cat("Best weights  :", result$best_weights, "\n")
cat("Results saved :", result$save_dir, "\n")
cat("Metrics output:", output_dir, "\n\n")

# ── What the numbers mean ─────────────────────────────────────
#
#  mAP@0.5      — main metric. > 0.5 = good. > 0.8 = great.
#  mAP@0.5:0.95 — strict metric. > 0.4 = good.
#  Recall       — % of real objects found. High = model finds them.
#  Precision    — % of detections that are correct.
#                 Low precision on tiny datasets is normal and
#                 improves significantly with more images.
#  F1           — harmonic mean of Precision and Recall.
#
#  If Precision is near 0 with only 5-10 images:
#   → Add more images (aim for 100+ per class)
#   → Train for more epochs (100-200)
#   → The model is NOT broken — it just needs more data.
