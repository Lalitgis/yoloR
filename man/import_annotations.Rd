\name{import_annotations}
\alias{import_annotations}
\alias{import_yolo_labels}
\alias{import_voc_xml}
\alias{import_coco_json}
\title{Import Already-Annotated Images}
\description{
Import images that have already been annotated with any tool
(LabelImg, CVAT, Label Studio, V7, Roboflow, etc.) into \pkg{yolor}
so they can be used for training immediately.

Four formats are supported:
\enumerate{
  \item \strong{YOLO TXT} — \code{.txt} label files (LabelImg YOLO mode,
    CVAT YOLO export, any YOLO-compatible tool)
  \item \strong{Pascal VOC XML} — per-image \code{.xml} files
    (LabelImg VOC mode, CVAT VOC export)
  \item \strong{COCO JSON} — single \code{_annotations.json} file
    (CVAT COCO export, Label Studio, etc.)
  \item \strong{Auto-detect} — \code{import_annotations()} inspects
    the folder and calls the right function automatically
}
All functions return a \code{shinylabel_dataset} object (same as
\code{\link{sl_read_db}}) or write a ready-to-train YOLO dataset
when \code{output_dir} is set.
}
\usage{
import_annotations(path, images_dir = NULL, class_names = NULL,
                   output_dir = NULL, val_split = 0.2,
                   seed = 42, copy_images = TRUE)

import_yolo_labels(images_dir, labels_dir = NULL,
                   class_names = NULL, output_dir = NULL,
                   val_split = 0.2, seed = 42,
                   copy_images = TRUE)

import_voc_xml(annotations_dir, images_dir = NULL,
               class_names = NULL, output_dir = NULL,
               val_split = 0.2, seed = 42,
               copy_images = TRUE)

import_coco_json(json_path, images_dir = NULL,
                 class_names = NULL, output_dir = NULL,
                 val_split = 0.2, seed = 42,
                 copy_images = TRUE)
}
\arguments{
  \item{path}{Path to the annotation folder (or \code{.json} file for COCO).
    \code{import_annotations()} auto-detects the format.}
  \item{images_dir}{Directory containing image files. If \code{NULL},
    the function looks for images next to the annotations.}
  \item{labels_dir}{Directory containing YOLO \code{.txt} label files.
    If \code{NULL}, looks for a \code{labels/} sibling folder or
    \code{.txt} files next to the images (LabelImg default).}
  \item{annotations_dir}{Directory containing VOC \code{.xml} files.}
  \item{json_path}{Path to a COCO JSON annotation file.}
  \item{class_names}{Character vector of class names in class-ID order,
    e.g. \code{c("cat", "dog", "bird")}. If \code{NULL}, names are
    auto-detected from \code{classes.txt}, \code{data.yaml}, or the
    annotation files.}
  \item{output_dir}{If set, a complete YOLO dataset (with train/val
    split and \code{data.yaml}) is written here and the path to
    \code{data.yaml} is returned — ready for \code{\link{yolo_train}}.}
  \item{val_split}{Fraction of images for validation (default \code{0.2}).}
  \item{seed}{Random seed for the train/val split (default \code{42}).}
  \item{copy_images}{Copy image files to \code{output_dir} (default
    \code{TRUE}). Set \code{FALSE} to skip copying.}
}
\value{
If \code{output_dir} is \code{NULL}: a \code{shinylabel_dataset} object
with \code{$images}, \code{$annotations}, \code{$classes}, and
\code{$summary}. You can pass this to \code{\link{sl_export_dataset}},
\code{\link{sl_class_summary}}, and \code{plot()}.

If \code{output_dir} is set: invisibly returns the path to
\code{data.yaml}, ready to pass directly to \code{\link{yolo_train}}.
}
\section{Auto-detection logic (import_annotations)}{
\code{import_annotations()} inspects the folder in this order:
\enumerate{
  \item \code{data.yaml} present → already a YOLO dataset, calls \code{\link{rf_load_yolo}}
  \item \code{.xml} files present → calls \code{\link{import_voc_xml}}
  \item \code{.json} files present → calls \code{\link{import_coco_json}}
  \item \code{.txt} files present → calls \code{\link{import_yolo_labels}}
  \item \code{labels/} sibling folder → calls \code{\link{import_yolo_labels}}
}
}
\section{Class name auto-detection}{
When \code{class_names = NULL}, the functions look for class names in:
\enumerate{
  \item \code{classes.txt} in the labels folder
  \item \code{classes.txt} in the images folder
  \item \code{data.yaml} in the parent folder
  \item Category names embedded in \code{.xml} or \code{.json} files
}
}
\examples{
\dontrun{
# ---- Auto-detect format (easiest) ----
yaml_path <- import_annotations(
  path       = "my_annotated_folder/",
  output_dir = "dataset/"
)
model  <- yolo_model("yolov8n")
result <- yolo_train(model, data = yaml_path, epochs = 50)


# ---- YOLO TXT labels (LabelImg YOLO mode) ----
# Standard layout:   images/ + labels/
yaml_path <- import_yolo_labels(
  images_dir  = "images/train/",
  labels_dir  = "labels/train/",
  class_names = c("cat", "dog", "bird"),
  output_dir  = "dataset/"
)

# LabelImg puts .txt next to images:
yaml_path <- import_yolo_labels(
  images_dir  = "labelimg_output/",    # contains both .jpg and .txt
  class_names = c("cat", "dog"),
  output_dir  = "dataset/"
)

# Auto-detect class names from classes.txt
yaml_path <- import_yolo_labels(
  images_dir = "images/",
  labels_dir = "labels/",
  output_dir = "dataset/"
)


# ---- Pascal VOC XML (LabelImg VOC mode, CVAT) ----
yaml_path <- import_voc_xml(
  annotations_dir = "voc_annotations/",
  images_dir      = "images/",
  class_names     = c("cat", "dog"),
  output_dir      = "dataset/"
)


# ---- COCO JSON (CVAT, Label Studio) ----
yaml_path <- import_coco_json(
  json_path  = "annotations/instances_train.json",
  images_dir = "images/train/",
  output_dir = "dataset/"
)


# ---- Inspect before exporting ----
ds <- import_yolo_labels(
  images_dir  = "images/",
  labels_dir  = "labels/",
  class_names = c("cat", "dog")
)
print(ds)
sl_class_summary(ds)
plot(ds)

# Then export when ready
sl_export_dataset(ds, output_dir = "dataset/", val_split = 0.2)
}
}
\seealso{
\code{\link{sl_read_db}}, \code{\link{sl_read_csv}},
\code{\link{rf_load_yolo}}, \code{\link{rf_coco_to_yolo}},
\code{\link{sl_export_dataset}}, \code{\link{yolo_train}}
}
