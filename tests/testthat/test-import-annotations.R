library(testthat)
library(yolor)

# ── Helpers ───────────────────────────────────────────────────

# Create a fake YOLO TXT dataset on disk
.make_yolo_txt_dataset <- function(classes = c("cat","dog"),
                                    n_images = 4) {
  dir <- tempfile()
  img_dir <- file.path(dir, "images")
  lbl_dir <- file.path(dir, "labels")
  dir.create(img_dir, recursive = TRUE)
  dir.create(lbl_dir, recursive = TRUE)

  # Write a minimal classes.txt
  writeLines(classes, file.path(lbl_dir, "classes.txt"))

  for (i in seq_len(n_images)) {
    fname <- sprintf("img%03d", i)
    # Tiny 10x10 white PNG via raw bytes
    png_path <- file.path(img_dir, paste0(fname, ".jpg"))
    writeLines("", png_path)   # placeholder — dims will be NA

    # Label: one box per image
    cls_id <- (i - 1L) %% length(classes)
    writeLines(
      sprintf("%d 0.500000 0.500000 0.200000 0.200000", cls_id),
      file.path(lbl_dir, paste0(fname, ".txt"))
    )
  }
  list(dir = dir, img_dir = img_dir, lbl_dir = lbl_dir,
       classes = classes, n = n_images)
}

# Create a fake VOC XML dataset
.make_voc_xml_dataset <- function(classes = c("cat","dog")) {
  dir <- tempfile()
  dir.create(dir, recursive = TRUE)

  for (i in 1:3) {
    fname <- sprintf("img%03d.jpg", i)
    writeLines("", file.path(dir, fname))

    label <- classes[((i - 1) %% length(classes)) + 1]
    xml_content <- sprintf('<?xml version="1.0"?>
<annotation>
  <filename>%s</filename>
  <size><width>640</width><height>480</height><depth>3</depth></size>
  <object>
    <name>%s</name>
    <bndbox>
      <xmin>10</xmin><ymin>10</ymin>
      <xmax>100</xmax><ymax>100</ymax>
    </bndbox>
  </object>
</annotation>', fname, label)
    writeLines(xml_content,
               file.path(dir, sprintf("img%03d.xml", i)))
  }
  list(dir = dir, classes = classes)
}

# Create a fake COCO JSON dataset
.make_coco_json_dataset <- function(classes = c("cat","dog")) {
  dir <- tempfile()
  dir.create(dir, recursive = TRUE)

  for (i in 1:3) {
    writeLines("", file.path(dir, sprintf("img%03d.jpg", i)))
  }

  coco <- list(
    images = list(
      list(id = 1L, file_name = "img001.jpg", width = 640L, height = 480L),
      list(id = 2L, file_name = "img002.jpg", width = 640L, height = 480L),
      list(id = 3L, file_name = "img003.jpg", width = 640L, height = 480L)
    ),
    categories = list(
      list(id = 1L, name = "cat"),
      list(id = 2L, name = "dog")
    ),
    annotations = list(
      list(id=1L, image_id=1L, category_id=1L,
           bbox=list(10,10,90,90), area=8100L, iscrowd=0L),
      list(id=2L, image_id=2L, category_id=2L,
           bbox=list(50,50,100,80), area=8000L, iscrowd=0L),
      list(id=3L, image_id=3L, category_id=1L,
           bbox=list(20,20,60,60), area=3600L, iscrowd=0L)
    )
  )

  json_path <- file.path(dir, "_annotations.coco.json")
  jsonlite::write_json(coco, json_path, auto_unbox = TRUE)
  list(dir = dir, json_path = json_path, classes = classes)
}

# ── import_yolo_labels ────────────────────────────────────────

test_that("import_yolo_labels errors on missing images_dir", {
  expect_error(import_yolo_labels("nonexistent/"), "not found")
})

test_that("import_yolo_labels returns shinylabel_dataset", {
  d  <- .make_yolo_txt_dataset()
  ds <- import_yolo_labels(d$img_dir, d$lbl_dir,
                            class_names = d$classes)
  expect_s3_class(ds, "shinylabel_dataset")
  expect_equal(nrow(ds$images), d$n)
  expect_equal(nrow(ds$annotations), d$n)
  expect_equal(nrow(ds$classes), length(d$classes))
})

test_that("import_yolo_labels auto-detects classes from classes.txt", {
  d  <- .make_yolo_txt_dataset()
  ds <- import_yolo_labels(d$img_dir, d$lbl_dir)  # no class_names
  expect_setequal(ds$classes$name, d$classes)
})

test_that("import_yolo_labels normalised coords are in [0,1]", {
  d  <- .make_yolo_txt_dataset()
  ds <- import_yolo_labels(d$img_dir, d$lbl_dir, d$classes)
  ann <- ds$annotations[!is.na(ds$annotations$x_center_norm), ]
  if (nrow(ann) > 0) {
    expect_true(all(ann$x_center_norm >= 0 & ann$x_center_norm <= 1))
    expect_true(all(ann$width_norm    >= 0 & ann$width_norm    <= 1))
  }
})

test_that("import_yolo_labels with output_dir returns yaml path", {
  d   <- .make_yolo_txt_dataset()
  out <- tempfile()
  yaml_path <- import_yolo_labels(d$img_dir, d$lbl_dir,
                                   d$classes, output_dir = out,
                                   val_split = 0.5, seed = 1,
                                   copy_images = FALSE)
  expect_true(file.exists(yaml_path))
  cfg <- yaml::read_yaml(yaml_path)
  expect_equal(cfg$nc, length(d$classes))
})

test_that("import_yolo_labels exported dataset passes validation", {
  d   <- .make_yolo_txt_dataset(n_images = 6)
  out <- tempfile()
  import_yolo_labels(d$img_dir, d$lbl_dir, d$classes,
                      output_dir = out, val_split = 0.33,
                      seed = 99, copy_images = FALSE)
  issues <- yolo_validate_dataset(out)
  expect_length(issues, 0)
})

# ── import_voc_xml ────────────────────────────────────────────

test_that("import_voc_xml errors without xml2", {
  skip_if(requireNamespace("xml2", quietly = TRUE),
          "xml2 is available — skipping no-xml2 test")
  d <- .make_voc_xml_dataset()
  expect_error(import_voc_xml(d$dir), "xml2")
})

test_that("import_voc_xml returns shinylabel_dataset", {
  skip_if_not_installed("xml2")
  d  <- .make_voc_xml_dataset()
  ds <- import_voc_xml(d$dir, d$dir)
  expect_s3_class(ds, "shinylabel_dataset")
  expect_equal(nrow(ds$images), 3)
  expect_equal(nrow(ds$annotations), 3)
  expect_setequal(ds$classes$name, d$classes)
})

test_that("import_voc_xml with output_dir returns yaml path", {
  skip_if_not_installed("xml2")
  d   <- .make_voc_xml_dataset()
  out <- tempfile()
  yaml_path <- import_voc_xml(d$dir, d$dir,
                               output_dir = out,
                               val_split  = 0.4,
                               copy_images = FALSE)
  expect_true(file.exists(yaml_path))
})

# ── import_coco_json ──────────────────────────────────────────

test_that("import_coco_json errors on missing file", {
  expect_error(import_coco_json("nofile.json"), "not found")
})

test_that("import_coco_json returns shinylabel_dataset", {
  d  <- .make_coco_json_dataset()
  ds <- import_coco_json(d$json_path, d$dir)
  expect_s3_class(ds, "shinylabel_dataset")
  expect_equal(nrow(ds$images), 3)
  expect_equal(nrow(ds$annotations), 3)
  expect_setequal(ds$classes$name, d$classes)
})

test_that("import_coco_json normalised coords valid", {
  d  <- .make_coco_json_dataset()
  ds <- import_coco_json(d$json_path, d$dir)
  ann <- ds$annotations[!is.na(ds$annotations$x_center_norm), ]
  if (nrow(ann) > 0) {
    expect_true(all(ann$x_center_norm >= 0 & ann$x_center_norm <= 1))
    expect_true(all(ann$y_center_norm >= 0 & ann$y_center_norm <= 1))
  }
})

test_that("import_coco_json with output_dir returns yaml path", {
  d   <- .make_coco_json_dataset()
  out <- tempfile()
  yaml_path <- import_coco_json(d$json_path, d$dir,
                                 output_dir  = out,
                                 val_split   = 0.4,
                                 copy_images = FALSE)
  expect_true(file.exists(yaml_path))
  cfg <- yaml::read_yaml(yaml_path)
  expect_equal(cfg$nc, 2)
})

# ── import_annotations (auto-detect) ─────────────────────────

test_that("import_annotations detects YOLO TXT format", {
  d  <- .make_yolo_txt_dataset()
  ds <- import_annotations(d$img_dir, images_dir = d$img_dir,
                            class_names = d$classes)
  expect_s3_class(ds, "shinylabel_dataset")
})

test_that("import_annotations detects COCO JSON format", {
  d  <- .make_coco_json_dataset()
  ds <- import_annotations(d$json_path, images_dir = d$dir)
  expect_s3_class(ds, "shinylabel_dataset")
})

test_that("import_annotations detects VOC XML format", {
  skip_if_not_installed("xml2")
  d  <- .make_voc_xml_dataset()
  ds <- import_annotations(d$dir, images_dir = d$dir)
  expect_s3_class(ds, "shinylabel_dataset")
})

test_that("import_annotations errors on unknown format", {
  empty <- tempfile(); dir.create(empty)
  expect_error(import_annotations(empty), "Could not detect")
})
