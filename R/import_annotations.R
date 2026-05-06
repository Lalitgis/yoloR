# ============================================================
#  import_annotations.R
#  Import already-annotated image datasets into yolor
#
#  Supported input formats:
#   1. YOLO TXT  — images/ + labels/ folders with .txt files
#                  (LabelImg, CVAT, Label Studio, Roboflow, etc.)
#   2. Pascal VOC XML — per-image .xml annotation files
#   3. COCO JSON — single _annotations.json file
#   4. Any folder where images and label files already exist
#
#  All paths converge to a shinylabel_dataset OR a ready-to-use
#  data.yaml so you can immediately call yolo_train().
# ============================================================

# ── 1. Import from YOLO TXT labels (most common) ─────────────

#' Import already-annotated images with YOLO TXT labels
#'
#' Use this when you already have a folder of images + matching
#' \code{.txt} label files (from LabelImg, CVAT, Label Studio,
#' Roboflow, V7, or any YOLO-compatible tool). The function reads
#' the labels, validates structure, and either returns a
#' \code{shinylabel_dataset} or a ready \code{data.yaml} path so
#' you can immediately call \code{\link{yolo_train}}.
#'
#' @param images_dir Directory containing image files
#'   (\code{.jpg}, \code{.jpeg}, \code{.png}, \code{.bmp},
#'   \code{.tif}).
#' @param labels_dir Directory containing YOLO \code{.txt} label
#'   files. Each file must match the image stem (e.g.
#'   \code{img001.jpg} → \code{img001.txt}). If \code{NULL}
#'   (default), looks for a \code{labels/} sibling folder next to
#'   \code{images_dir}, or a \code{labels/} subfolder inside
#'   \code{images_dir}.
#' @param class_names Character vector of class names in class-ID
#'   order, e.g. \code{c("cat", "dog", "bird")}. If \code{NULL}
#'   and a \code{data.yaml} or \code{classes.txt} is found next to
#'   the images, class names are read from there automatically.
#' @param output_dir If provided, exports a ready-to-train YOLO
#'   dataset (with train/val split and \code{data.yaml}) to this
#'   directory. You can then pass \code{output_dir/data.yaml}
#'   directly to \code{\link{yolo_train}}.
#' @param val_split Validation fraction when \code{output_dir} is
#'   set (default \code{0.2}).
#' @param seed Random seed for train/val split (default \code{42}).
#' @param copy_images Copy images to \code{output_dir} (default
#'   \code{TRUE}). Set \code{FALSE} to use symlinks / skip copying.
#'
#' @return If \code{output_dir} is \code{NULL}: a
#'   \code{shinylabel_dataset} object.\cr
#'   If \code{output_dir} is set: invisibly returns the path to
#'   \code{data.yaml}, ready for \code{\link{yolo_train}}.
#'
#' @examples
#' \dontrun{
#' # ---- Minimal usage — just images + labels folders ----
#' ds <- import_yolo_labels(
#'   images_dir  = "my_images/",
#'   labels_dir  = "my_labels/",
#'   class_names = c("cat", "dog", "bird")
#' )
#' print(ds)
#'
#' # ---- Export and train immediately ----
#' yaml_path <- import_yolo_labels(
#'   images_dir  = "my_images/",
#'   labels_dir  = "my_labels/",
#'   class_names = c("cat", "dog"),
#'   output_dir  = "dataset/",
#'   val_split   = 0.2
#' )
#' model  <- yolo_model("yolov8n")
#' result <- yolo_train(model, data = yaml_path, epochs = 50)
#'
#' # ---- Auto-detect classes from classes.txt or data.yaml ----
#' yaml_path <- import_yolo_labels(
#'   images_dir = "my_images/",
#'   output_dir = "dataset/"
#' )
#'
#' # ---- LabelImg export (saves labels next to images) ----
#' yaml_path <- import_yolo_labels(
#'   images_dir  = "labelimg_output/",
#'   class_names = c("person", "car", "truck"),
#'   output_dir  = "dataset/"
#' )
#' }
#'
#' @seealso \code{\link{import_voc_xml}}, \code{\link{import_coco_json}},
#'   \code{\link{yolo_train}}, \code{\link{sl_export_dataset}}
#' @export
import_yolo_labels <- function(images_dir,
                                labels_dir  = NULL,
                                class_names = NULL,
                                output_dir  = NULL,
                                val_split   = 0.2,
                                seed        = 42,
                                copy_images = TRUE) {

  images_dir <- fs::path_abs(images_dir)
  if (!fs::dir_exists(images_dir))
    abort(glue("images_dir not found: {images_dir}"))

  # ── Resolve labels directory ────────────────────────────────
  labels_dir <- .resolve_labels_dir(images_dir, labels_dir)

  # ── Auto-detect class names ─────────────────────────────────
  if (is.null(class_names)) {
    class_names <- .detect_class_names(images_dir, labels_dir)
    if (is.null(class_names)) {
      abort(c(
        "Could not detect class names automatically.",
        "i" = "Provide class_names = c('cat', 'dog', ...) explicitly.",
        "i" = "Or place a classes.txt or data.yaml next to your images."
      ))
    }
    cli::cli_alert_info(
      "Auto-detected classes: {paste(class_names, collapse=', ')}"
    )
  }

  # ── Scan image files ────────────────────────────────────────
  img_exts  <- "\\.(jpg|jpeg|png|bmp|tif|tiff)$"
  img_files <- fs::dir_ls(images_dir, regexp = img_exts,
                           ignore.case = TRUE, recurse = FALSE)

  if (length(img_files) == 0)
    abort(glue("No image files found in: {images_dir}"))

  cli::cli_alert_info(
    "Found {length(img_files)} images in: {images_dir}"
  )

  # ── Read label files and build dataset ──────────────────────
  class_map <- stats::setNames(seq_along(class_names) - 1L, class_names)

  pb <- cli::cli_progress_bar("Reading labels",
                               total = length(img_files))

  img_rows <- list()
  ann_rows <- list()
  img_id   <- 0L

  for (img_path in img_files) {
    img_id   <- img_id + 1L
    img_name <- fs::path_file(img_path)
    stem     <- fs::path_ext_remove(img_name)
    lbl_path <- fs::path(labels_dir, paste0(stem, ".txt"))

    # Image dimensions
    dims <- .safe_image_dims(img_path)

    img_rows[[img_id]] <- data.frame(
      id       = img_id,
      filepath = as.character(img_path),
      width    = dims$w,
      height   = dims$h,
      status   = "done",
      stringsAsFactors = FALSE
    )

    # Parse label file
    if (!fs::file_exists(lbl_path)) {
      cli::cli_progress_update(id = pb)
      next
    }

    lines <- readLines(lbl_path, warn = FALSE)
    lines <- trimws(lines[nzchar(trimws(lines))])

    for (line in lines) {
      parts <- strsplit(line, "\\s+")[[1]]
      if (length(parts) < 5) next

      cls_id  <- as.integer(parts[1])
      xc_norm <- as.numeric(parts[2])
      yc_norm <- as.numeric(parts[3])
      w_norm  <- as.numeric(parts[4])
      h_norm  <- as.numeric(parts[5])

      cls_name <- if (cls_id < length(class_names))
        class_names[cls_id + 1L] else as.character(cls_id)

      ann_rows[[length(ann_rows) + 1L]] <- data.frame(
        id            = length(ann_rows) + 1L,
        image_id      = img_id,
        class_id      = cls_id,
        class_name    = cls_name,
        class_color   = NA_character_,
        x_center_norm = xc_norm,
        y_center_norm = yc_norm,
        width_norm    = w_norm,
        height_norm   = h_norm,
        xmin = (xc_norm - w_norm / 2) * dims$w,
        ymin = (yc_norm - h_norm / 2) * dims$h,
        xmax = (xc_norm + w_norm / 2) * dims$w,
        ymax = (yc_norm + h_norm / 2) * dims$h,
        filepath      = as.character(img_path),
        img_width     = dims$w,
        img_height    = dims$h,
        stringsAsFactors = FALSE
      )
    }
    cli::cli_progress_update(id = pb)
  }
  cli::cli_progress_done(id = pb)

  images <- tibble::as_tibble(dplyr::bind_rows(img_rows))
  annotations <- if (length(ann_rows) > 0)
    tibble::as_tibble(dplyr::bind_rows(ann_rows))
  else
    tibble::tibble()

  classes <- tibble::tibble(
    class_id = seq_along(class_names) - 1L,
    name     = class_names,
    color    = NA_character_
  )

  n_labeled   <- sum(vapply(img_rows, function(r) {
    any(vapply(ann_rows, function(a) a$image_id == r$id, logical(1)))
  }, logical(1)))

  cli::cli_alert_success(glue(
    "Loaded: {nrow(images)} images | ",
    "{nrow(annotations)} boxes | ",
    "{length(class_names)} classes"
  ))

  ds <- .make_shinylabel_dataset(
    images, annotations, classes,
    source = "yolo_txt", path = as.character(images_dir)
  )

  # ── Export to YOLO layout if requested ──────────────────────
  if (!is.null(output_dir)) {
    yaml_path <- sl_export_dataset(
      ds,
      output_dir  = output_dir,
      val_split   = val_split,
      seed        = seed,
      copy_images = copy_images
    )
    return(invisible(yaml_path))
  }

  ds
}


# ── 2. Import from Pascal VOC XML ────────────────────────────

#' Import already-annotated images with Pascal VOC XML labels
#'
#' Reads Pascal VOC \code{.xml} annotation files (as exported by
#' LabelImg in VOC mode, CVAT, or similar tools). Converts pixel
#' bounding boxes to normalised YOLO format and returns a
#' \code{shinylabel_dataset}.
#'
#' @param annotations_dir Directory containing the \code{.xml} files.
#'   Each \code{.xml} must contain a \code{<filename>} tag pointing
#'   to the image and \code{<object>} tags with \code{<bndbox>}.
#' @param images_dir Directory containing the images. If \code{NULL},
#'   uses \code{annotations_dir}.
#' @param class_names Optional character vector to fix class ordering.
#'   If \code{NULL}, class IDs are assigned alphabetically from the
#'   labels found in the XML files.
#' @param output_dir If provided, exports a ready-to-train YOLO dataset.
#' @param val_split Validation fraction (default \code{0.2}).
#' @param seed Random seed (default \code{42}).
#' @param copy_images Copy images to \code{output_dir} (default \code{TRUE}).
#'
#' @return A \code{shinylabel_dataset}, or invisibly the \code{data.yaml}
#'   path if \code{output_dir} is set.
#'
#' @examples
#' \dontrun{
#' # LabelImg VOC format export
#' yaml_path <- import_voc_xml(
#'   annotations_dir = "labelimg_voc_output/",
#'   images_dir      = "my_images/",
#'   class_names     = c("cat", "dog"),
#'   output_dir      = "dataset/"
#' )
#' model  <- yolo_model("yolov8n")
#' result <- yolo_train(model, data = yaml_path, epochs = 50)
#' }
#'
#' @seealso \code{\link{import_yolo_labels}}, \code{\link{import_coco_json}}
#' @export
import_voc_xml <- function(annotations_dir,
                            images_dir  = NULL,
                            class_names = NULL,
                            output_dir  = NULL,
                            val_split   = 0.2,
                            seed        = 42,
                            copy_images = TRUE) {

  if (!requireNamespace("xml2", quietly = TRUE))
    abort("Package 'xml2' is required. Install it: install.packages('xml2')")

  annotations_dir <- fs::path_abs(annotations_dir)
  if (!fs::dir_exists(annotations_dir))
    abort(glue("annotations_dir not found: {annotations_dir}"))

  images_dir <- if (is.null(images_dir)) annotations_dir else
    fs::path_abs(images_dir)

  xml_files <- fs::dir_ls(annotations_dir, regexp = "\\.xml$",
                           ignore.case = TRUE)
  if (length(xml_files) == 0)
    abort(glue("No .xml files found in: {annotations_dir}"))

  cli::cli_alert_info("Found {length(xml_files)} VOC XML files")

  # First pass — collect all class names if not provided
  if (is.null(class_names)) {
    all_labels <- character(0)
    for (xf in xml_files) {
      doc  <- xml2::read_xml(xf)
      objs <- xml2::xml_find_all(doc, ".//object")
      lbls <- xml2::xml_text(xml2::xml_find_first(objs, ".//name"))
      all_labels <- union(all_labels, lbls)
    }
    class_names <- sort(all_labels)
    cli::cli_alert_info(
      "Auto-detected classes: {paste(class_names, collapse=', ')}"
    )
  }

  class_map <- stats::setNames(seq_along(class_names) - 1L, class_names)

  img_rows <- list()
  ann_rows <- list()
  img_id   <- 0L

  pb <- cli::cli_progress_bar("Reading VOC XML", total = length(xml_files))

  for (xf in xml_files) {
    doc      <- xml2::read_xml(xf)
    fname    <- xml2::xml_text(xml2::xml_find_first(doc, ".//filename"))
    img_path <- fs::path(images_dir, fname)

    # Try to find image if not at expected path
    if (!fs::file_exists(img_path)) {
      found <- fs::dir_ls(images_dir,
                           regexp = paste0(fs::path_ext_remove(fname), "\\."),
                           ignore.case = TRUE)
      if (length(found) > 0) img_path <- found[1]
    }

    # Image dimensions from XML or file
    w <- as.integer(xml2::xml_text(
      xml2::xml_find_first(doc, ".//size/width")))
    h <- as.integer(xml2::xml_text(
      xml2::xml_find_first(doc, ".//size/height")))

    if (is.na(w) || is.na(h) || w == 0 || h == 0) {
      dims <- .safe_image_dims(img_path)
      w <- dims$w; h <- dims$h
    }

    img_id <- img_id + 1L
    img_rows[[img_id]] <- data.frame(
      id = img_id, filepath = as.character(img_path),
      width = w, height = h, status = "done",
      stringsAsFactors = FALSE
    )

    objs <- xml2::xml_find_all(doc, ".//object")
    for (obj in objs) {
      label  <- xml2::xml_text(xml2::xml_find_first(obj, ".//name"))
      cls_id <- class_map[[label]]
      if (is.null(cls_id)) { cls_id <- 0L }

      xmin <- as.numeric(xml2::xml_text(
        xml2::xml_find_first(obj, ".//bndbox/xmin")))
      ymin <- as.numeric(xml2::xml_text(
        xml2::xml_find_first(obj, ".//bndbox/ymin")))
      xmax <- as.numeric(xml2::xml_text(
        xml2::xml_find_first(obj, ".//bndbox/xmax")))
      ymax <- as.numeric(xml2::xml_text(
        xml2::xml_find_first(obj, ".//bndbox/ymax")))

      xc <- (xmin + xmax) / (2 * w)
      yc <- (ymin + ymax) / (2 * h)
      bw <- (xmax - xmin) / w
      bh <- (ymax - ymin) / h

      ann_rows[[length(ann_rows) + 1L]] <- data.frame(
        id            = length(ann_rows) + 1L,
        image_id      = img_id,
        class_id      = as.integer(cls_id),
        class_name    = label,
        class_color   = NA_character_,
        x_center_norm = xc, y_center_norm = yc,
        width_norm    = bw, height_norm   = bh,
        xmin = xmin, ymin = ymin, xmax = xmax, ymax = ymax,
        filepath   = as.character(img_path),
        img_width  = w, img_height = h,
        stringsAsFactors = FALSE
      )
    }
    cli::cli_progress_update(id = pb)
  }
  cli::cli_progress_done(id = pb)

  images      <- tibble::as_tibble(dplyr::bind_rows(img_rows))
  annotations <- tibble::as_tibble(dplyr::bind_rows(ann_rows))
  classes     <- tibble::tibble(class_id = seq_along(class_names) - 1L,
                                 name     = class_names,
                                 color    = NA_character_)

  cli::cli_alert_success(glue(
    "Loaded: {nrow(images)} images | {nrow(annotations)} boxes | ",
    "{length(class_names)} classes"
  ))

  ds <- .make_shinylabel_dataset(images, annotations, classes,
                                  source = "voc_xml",
                                  path   = as.character(annotations_dir))

  if (!is.null(output_dir)) {
    return(invisible(sl_export_dataset(ds, output_dir, val_split,
                                        seed, copy_images)))
  }
  ds
}


# ── 3. Import from COCO JSON ──────────────────────────────────

#' Import already-annotated images with COCO JSON labels
#'
#' Reads a COCO-format \code{_annotations.json} (or any COCO JSON)
#' and converts to a \code{shinylabel_dataset}. Supports bounding
#' box annotations (\code{bbox} field).
#'
#' @param json_path Path to the COCO JSON annotation file.
#' @param images_dir Directory containing the images referenced by
#'   the JSON. If \code{NULL}, uses the folder containing the JSON.
#' @param class_names Optional character vector to fix class ordering.
#' @param output_dir If provided, exports a ready-to-train YOLO dataset.
#' @param val_split Validation fraction (default \code{0.2}).
#' @param seed Random seed (default \code{42}).
#' @param copy_images Copy images to \code{output_dir} (default \code{TRUE}).
#'
#' @return A \code{shinylabel_dataset}, or invisibly the \code{data.yaml}
#'   path if \code{output_dir} is set.
#'
#' @examples
#' \dontrun{
#' # CVAT COCO export
#' yaml_path <- import_coco_json(
#'   json_path  = "annotations/instances_train.json",
#'   images_dir = "images/train/",
#'   output_dir = "dataset/"
#' )
#' model  <- yolo_model("yolov8n")
#' result <- yolo_train(model, data = yaml_path, epochs = 50)
#' }
#'
#' @seealso \code{\link{import_yolo_labels}}, \code{\link{import_voc_xml}}
#' @export
import_coco_json <- function(json_path,
                              images_dir  = NULL,
                              class_names = NULL,
                              output_dir  = NULL,
                              val_split   = 0.2,
                              seed        = 42,
                              copy_images = TRUE) {

  if (!file.exists(json_path))
    abort(glue("JSON file not found: {json_path}"))

  images_dir <- if (is.null(images_dir))
    fs::path_dir(json_path) else fs::path_abs(images_dir)

  cli::cli_alert_info("Reading COCO JSON: {json_path}")
  coco <- jsonlite::fromJSON(json_path, simplifyVector = FALSE)

  # Build category map (COCO IDs are 1-based)
  cats      <- coco$categories
  cat_ids   <- vapply(cats, `[[`, integer(1),   "id")
  cat_names <- vapply(cats, `[[`, character(1), "name")

  if (is.null(class_names)) {
    class_names <- cat_names[order(cat_ids)]
  }
  # Map COCO category_id → 0-based YOLO class_id
  coco_to_yolo <- stats::setNames(
    seq_along(cat_ids) - 1L, as.character(cat_ids)
  )
  coco_to_name <- stats::setNames(cat_names, as.character(cat_ids))

  # Image lookup
  img_lookup <- stats::setNames(
    lapply(coco$images, function(im)
      list(file_name = im$file_name,
           width     = as.integer(im$width),
           height    = as.integer(im$height))),
    vapply(coco$images, function(im) as.character(im$id), character(1))
  )

  # Group annotations by image_id
  ann_by_img <- list()
  for (ann in coco$annotations) {
    key <- as.character(ann$image_id)
    ann_by_img[[key]] <- c(ann_by_img[[key]], list(ann))
  }

  img_rows <- list()
  ann_rows <- list()

  pb <- cli::cli_progress_bar("Reading COCO JSON",
                               total = length(coco$images))

  for (i in seq_along(coco$images)) {
    im      <- coco$images[[i]]
    img_id  <- as.character(im$id)
    fname   <- im$file_name
    w       <- as.integer(im$width)
    h       <- as.integer(im$height)

    img_path <- fs::path(images_dir, fname)
    if (!fs::file_exists(img_path)) {
      img_path <- fs::path(images_dir, fs::path_file(fname))
    }
    if (is.na(w) || w == 0) {
      dims <- .safe_image_dims(img_path); w <- dims$w; h <- dims$h
    }

    img_rows[[i]] <- data.frame(
      id = i, filepath = as.character(img_path),
      width = w, height = h, status = "done",
      stringsAsFactors = FALSE
    )

    anns <- ann_by_img[[img_id]]
    for (ann in anns) {
      bbox     <- ann$bbox   # [x_tl, y_tl, width, height]
      cat_id   <- as.character(ann$category_id)
      cls_id   <- coco_to_yolo[[cat_id]]
      cls_name <- coco_to_name[[cat_id]]
      if (is.null(cls_id)) next

      xc <- (bbox[[1]] + bbox[[3]] / 2) / w
      yc <- (bbox[[2]] + bbox[[4]] / 2) / h
      bw <- bbox[[3]] / w
      bh <- bbox[[4]] / h

      ann_rows[[length(ann_rows) + 1L]] <- data.frame(
        id            = length(ann_rows) + 1L,
        image_id      = i,
        class_id      = as.integer(cls_id),
        class_name    = cls_name,
        class_color   = NA_character_,
        x_center_norm = xc, y_center_norm = yc,
        width_norm    = bw, height_norm   = bh,
        xmin = bbox[[1]], ymin = bbox[[2]],
        xmax = bbox[[1]] + bbox[[3]], ymax = bbox[[2]] + bbox[[4]],
        filepath   = as.character(img_path),
        img_width  = w, img_height = h,
        stringsAsFactors = FALSE
      )
    }
    cli::cli_progress_update(id = pb)
  }
  cli::cli_progress_done(id = pb)

  images      <- tibble::as_tibble(dplyr::bind_rows(img_rows))
  annotations <- tibble::as_tibble(dplyr::bind_rows(ann_rows))
  classes     <- tibble::tibble(class_id = seq_along(class_names) - 1L,
                                 name     = class_names,
                                 color    = NA_character_)

  cli::cli_alert_success(glue(
    "Loaded: {nrow(images)} images | {nrow(annotations)} boxes | ",
    "{length(class_names)} classes"
  ))

  ds <- .make_shinylabel_dataset(images, annotations, classes,
                                  source = "coco_json",
                                  path   = json_path)

  if (!is.null(output_dir)) {
    return(invisible(sl_export_dataset(ds, output_dir, val_split,
                                        seed, copy_images)))
  }
  ds
}


# ── 4. Auto-detect any format ─────────────────────────────────

#' Import annotated images — auto-detect format
#'
#' Inspects the provided folder and automatically calls the right
#' import function based on what it finds:
#' \itemize{
#'   \item \code{.txt} label files → \code{\link{import_yolo_labels}}
#'   \item \code{.xml} files → \code{\link{import_voc_xml}}
#'   \item \code{.json} file → \code{\link{import_coco_json}}
#'   \item \code{data.yaml} present → \code{\link{rf_load_yolo}}
#' }
#'
#' @param path Path to the annotation folder (or JSON file for COCO).
#' @param images_dir Optional. Images folder when annotations are
#'   in a separate directory.
#' @param class_names Optional class names vector.
#' @param output_dir If set, exports a ready-to-train YOLO dataset here.
#' @param val_split Validation fraction (default \code{0.2}).
#' @param seed Random seed (default \code{42}).
#' @param copy_images Copy images to output (default \code{TRUE}).
#'
#' @return A \code{shinylabel_dataset} or \code{data.yaml} path.
#'
#' @examples
#' \dontrun{
#' # Let yolor figure out the format automatically
#' yaml_path <- import_annotations(
#'   path       = "my_annotated_folder/",
#'   output_dir = "dataset/"
#' )
#' model  <- yolo_model("yolov8n")
#' result <- yolo_train(model, data = yaml_path, epochs = 50)
#' }
#'
#' @export
import_annotations <- function(path,
                                images_dir  = NULL,
                                class_names = NULL,
                                output_dir  = NULL,
                                val_split   = 0.2,
                                seed        = 42,
                                copy_images = TRUE) {

  path <- fs::path_abs(path)

  # COCO JSON file passed directly
  if (fs::is_file(path) && grepl("\\.json$", path, ignore.case = TRUE)) {
    cli::cli_alert_info("Detected format: COCO JSON")
    return(import_coco_json(path, images_dir, class_names,
                             output_dir, val_split, seed, copy_images))
  }

  if (!fs::is_dir(path)) abort(glue("Path not found: {path}"))

  # data.yaml present → already a YOLO dataset
  if (fs::file_exists(fs::path(path, "data.yaml"))) {
    cli::cli_alert_info("Detected format: YOLO dataset (data.yaml found)")
    return(rf_load_yolo(path))
  }

  # VOC XML files
  xml_files <- fs::dir_ls(path, regexp = "\\.xml$", ignore.case = TRUE)
  if (length(xml_files) > 0) {
    cli::cli_alert_info("Detected format: Pascal VOC XML ({length(xml_files)} files)")
    return(import_voc_xml(path, images_dir %||% path, class_names,
                           output_dir, val_split, seed, copy_images))
  }

  # COCO JSON inside folder
  json_files <- fs::dir_ls(path, regexp = "\\.json$", ignore.case = TRUE)
  if (length(json_files) > 0) {
    cli::cli_alert_info("Detected format: COCO JSON ({fs::path_file(json_files[1])})")
    return(import_coco_json(as.character(json_files[1]),
                             images_dir %||% path, class_names,
                             output_dir, val_split, seed, copy_images))
  }

  # YOLO TXT labels
  txt_files <- fs::dir_ls(path, regexp = "\\.txt$", ignore.case = TRUE)
  if (length(txt_files) > 0) {
    cli::cli_alert_info("Detected format: YOLO TXT ({length(txt_files)} label files)")
    return(import_yolo_labels(path, labels_dir = path, class_names,
                               output_dir, val_split, seed, copy_images))
  }

  # Check for labels/ sibling
  labels_sib <- fs::path(fs::path_dir(path), "labels",
                           fs::path_file(path))
  if (fs::dir_exists(labels_sib)) {
    cli::cli_alert_info("Detected format: YOLO TXT (labels/ sibling folder)")
    return(import_yolo_labels(path, labels_sib, class_names,
                               output_dir, val_split, seed, copy_images))
  }

  abort(c(
    "Could not detect annotation format in: {path}",
    "i" = "Expected one of: .txt (YOLO), .xml (VOC), .json (COCO), or data.yaml",
    "i" = "Use import_yolo_labels(), import_voc_xml(), or import_coco_json() directly."
  ))
}


# ── Internal helpers ──────────────────────────────────────────

#' @keywords internal
.resolve_labels_dir <- function(images_dir, labels_dir) {
  if (!is.null(labels_dir)) {
    labels_dir <- fs::path_abs(labels_dir)
    if (!fs::dir_exists(labels_dir))
      abort(glue("labels_dir not found: {labels_dir}"))
    return(labels_dir)
  }

  # Check for labels/ sibling (standard YOLO layout)
  parent     <- fs::path_dir(images_dir)
  sibling    <- fs::path(parent, "labels",
                          fs::path_file(images_dir))
  if (fs::dir_exists(sibling)) return(sibling)

  # labels/ sibling at parent level
  sibling2 <- fs::path(parent, "labels")
  if (fs::dir_exists(sibling2)) return(sibling2)

  # Labels in same folder as images (LabelImg default)
  cli::cli_alert_info(
    "labels_dir not found — looking for .txt files in images_dir"
  )
  images_dir
}

#' @keywords internal
.detect_class_names <- function(images_dir, labels_dir) {
  # 1. classes.txt in labels_dir
  ct <- fs::path(labels_dir, "classes.txt")
  if (fs::file_exists(ct)) {
    nms <- readLines(ct, warn = FALSE)
    nms <- trimws(nms[nzchar(trimws(nms))])
    if (length(nms) > 0) return(nms)
  }

  # 2. classes.txt in images_dir
  ct2 <- fs::path(images_dir, "classes.txt")
  if (fs::file_exists(ct2)) {
    nms <- readLines(ct2, warn = FALSE)
    nms <- trimws(nms[nzchar(trimws(nms))])
    if (length(nms) > 0) return(nms)
  }

  # 3. data.yaml in parent
  for (candidate in c(
    fs::path(fs::path_dir(images_dir), "data.yaml"),
    fs::path(images_dir, "data.yaml"),
    fs::path(labels_dir, "data.yaml")
  )) {
    if (fs::file_exists(candidate)) {
      cfg  <- yaml::read_yaml(candidate)
      nms  <- unlist(cfg$names)
      if (!is.null(nms) && length(nms) > 0) return(nms)
    }
  }
  NULL
}

#' @keywords internal
.safe_image_dims <- function(img_path) {
  if (!fs::file_exists(img_path))
    return(list(w = NA_integer_, h = NA_integer_))
  tryCatch({
    info <- magick::image_info(magick::image_read(img_path))
    list(w = as.integer(info$width), h = as.integer(info$height))
  }, error = function(e) list(w = NA_integer_, h = NA_integer_))
}
