"""Finds candidate food regions on a plate with YOLOv8 (Open Images V7).

    python src/detect.py path/to/plate_photo.jpg
    python src/detect.py photo.jpg --conf 0.15     # catch weaker detections

Class-agnostic by design: YOLO only has to separate regions — the classifier
names them — and Open Images has no label for most dishes, so every detection
is kept except an explicit non-food blocklist. Post-processing then applies
cross-class NMS (one box per region) and drops container boxes (a whole plate
or tray) that hold several smaller ones.
"""
import argparse

import cv2
from PIL import Image
from ultralytics import YOLO

# Open Images labels that are never edible. Everything NOT listed here counts
# as a candidate region — the dishes we care about have no labels to allowlist.
NON_FOOD = {
    # people
    "Person", "Man", "Woman", "Boy", "Girl", "Human face", "Human hand",
    "Human arm", "Human body", "Human head", "Human hair", "Human eye",
    "Human nose", "Human mouth", "Human ear", "Human leg", "Human foot",
    "Human beard", "Glasses", "Watch",
    # furniture / surfaces
    "Table", "Kitchen & dining room table", "Coffee table", "Desk", "Chair",
    "Furniture", "Countertop", "Cabinetry", "Shelf", "Couch", "Bench",
    "Cupboard", "Drawer", "Stool", "Nightstand",
    # cutlery / utensils (NOT plates/bowls -- those usually frame the food)
    "Fork", "Spoon", "Knife", "Kitchen knife", "Chopsticks", "Ladle",
    "Spatula", "Whisk", "Kitchen utensil", "Tableware", "Kitchenware",
    "Cutting board", "Frying pan", "Wok", "Mixing bowl", "Measuring cup",
    "Can opener", "Pizza cutter", "Salt and pepper shakers",
    # drinks & drinkware -- calories come from the plate, not the glass
    "Bottle", "Wine glass", "Coffee cup", "Mug", "Jug", "Teapot", "Kettle",
    "Drink", "Beer", "Wine", "Coffee", "Tea", "Juice", "Milk", "Cocktail",
    "Drinking straw", "Tin can",
    # appliances / electronics
    "Mobile phone", "Laptop", "Computer monitor", "Television",
    "Microwave oven", "Oven", "Refrigerator", "Toaster", "Blender",
    "Gas stove", "Home appliance", "Kitchen appliance", "Sink", "Tap",
    "Dishwasher", "Food processor",
    # room / background
    "Window", "Door", "Wall clock", "Picture frame", "Poster", "Mirror",
    "Curtain", "Lamp", "Light bulb", "Candle", "Ceiling fan",
    "Houseplant", "Plant", "Flower", "Flowerpot", "Tree",
    # misc objects
    "Book", "Pen", "Paper towel", "Facial tissue holder",
    "Handbag", "Backpack", "Box", "Plastic bag", "Luggage and bags",
    "Clothing", "Shirt", "Jeans", "Dress", "Suit", "Jacket", "Hat",
    "Dog", "Cat", "Bird", "Car", "Bicycle", "Vehicle", "Toy",
}


def iou(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def containment(inner, outer):
    """Fraction of `inner` that sits inside `outer` (0-1)."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_inner = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / area_inner if area_inner else 0.0


def cross_class_nms(dets, iou_thresh=0.55):
    """YOLO often fires 'Food' + 'Salad' + 'Bowl' on the same region.
    Keep only the highest-confidence box per region."""
    kept = []
    for det in sorted(dets, key=lambda d: -d["confidence"]):
        if all(iou(det["box"], k["box"]) < iou_thresh for k in kept):
            kept.append(det)
    return kept


def drop_container_boxes(dets, img_area, cover_thresh=0.55, min_children=2):
    """If a big box (the whole plate/tray) contains >=2 smaller boxes, drop
    the big one -- we want the individual items, not the tray."""
    if len(dets) < min_children + 1:
        return dets

    keep = []
    for i, det in enumerate(dets):
        box = det["box"]
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area / img_area < cover_thresh:
            keep.append(det)
            continue
        children = sum(
            1 for j, other in enumerate(dets)
            if j != i and containment(other["box"], box) > 0.8
        )
        if children < min_children:
            keep.append(det)
    return keep or dets  # never return nothing


_yolo_cache = {}


def _get_yolo(weights):
    """Load a YOLO model once per weights file and reuse it across calls."""
    if weights not in _yolo_cache:
        _yolo_cache[weights] = YOLO(weights)
    return _yolo_cache[weights]


def detect_food(image_path, conf_threshold=0.1, weights="yolov8m-oiv7.pt",
                iou_thresh=0.45, max_det=50, agnostic_nms=False,
                min_area_frac=0.02, debug=False):
    """Return the candidate food regions in an image, each with its PIL crop."""
    model = _get_yolo(weights)
    results = model(image_path, conf=conf_threshold, iou=iou_thresh,
                    max_det=max_det, agnostic_nms=agnostic_nms,
                    verbose=False)[0]

    img = Image.open(image_path).convert("RGB")
    img_area = img.width * img.height

    raw = []
    for box in results.boxes:
        label = results.names[int(box.cls)]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        raw.append({
            "yolo_label": label,
            "confidence": round(float(box.conf), 3),
            "box": (x1, y1, x2, y2),
        })

    if debug:
        print(f"\n{'='*60}")
        print(f"STAGE 0 — raw YOLO output: {len(raw)} box(es)")
        print(f"{'='*60}")
        for d in sorted(raw, key=lambda x: -x["confidence"]):
            x1, y1, x2, y2 = d["box"]
            print(f"  {d['yolo_label']:<30} conf={d['confidence']:<6} "
                  f"box=({x1},{y1},{x2},{y2})  size={x2-x1}x{y2-y1}")

    after_blocklist = []
    blocklist_removed = []
    for d in raw:
        x1, y1, x2, y2 = d["box"]
        if d["yolo_label"] in NON_FOOD:
            blocklist_removed.append(d)
        elif x2 <= x1 or y2 <= y1:
            blocklist_removed.append(d)
        else:
            after_blocklist.append(d)

    if debug:
        print(f"\n{'='*60}")
        print(f"STAGE 1 — after NON_FOOD blocklist: "
              f"{len(after_blocklist)} kept, {len(blocklist_removed)} removed")
        print(f"{'='*60}")
        if blocklist_removed:
            print("  Removed:")
            for d in blocklist_removed:
                print(f"    {d['yolo_label']:<30} conf={d['confidence']}")
        print("  Kept:")
        for d in after_blocklist:
            x1, y1, x2, y2 = d["box"]
            print(f"    {d['yolo_label']:<30} conf={d['confidence']:<6} "
                  f"box=({x1},{y1},{x2},{y2})")

    min_px = img_area * min_area_frac
    after_area = [d for d in after_blocklist
                  if (d["box"][2]-d["box"][0]) * (d["box"][3]-d["box"][1]) >= min_px]
    area_removed = [d for d in after_blocklist if d not in after_area]

    if debug:
        print(f"\n{'='*60}")
        print(f"STAGE 1b — after min-area filter ({min_area_frac*100:.1f}% of image = "
              f"{int(min_px)}px²): {len(after_area)} kept, {len(area_removed)} removed")
        print(f"{'='*60}")
        if area_removed:
            print("  Removed (too small):")
            for d in area_removed:
                x1, y1, x2, y2 = d["box"]
                pct = (x2-x1)*(y2-y1)/img_area*100
                print(f"    {d['yolo_label']:<30} conf={d['confidence']:<6} "
                      f"size={x2-x1}x{y2-y1} ({pct:.2f}% of image)")
        if after_area:
            print("  Kept:")
            for d in after_area:
                x1, y1, x2, y2 = d["box"]
                print(f"    {d['yolo_label']:<30} conf={d['confidence']:<6} "
                      f"box=({x1},{y1},{x2},{y2})")

    after_nms = cross_class_nms(after_area)
    nms_removed = [d for d in after_area if d not in after_nms]

    if debug:
        print(f"\n{'='*60}")
        print(f"STAGE 2 — after cross-class NMS (iou_thresh=0.55): "
              f"{len(after_nms)} kept, {len(nms_removed)} removed")
        print(f"{'='*60}")
        if nms_removed:
            print("  Removed (overlapped a higher-confidence box):")
            for d in nms_removed:
                x1, y1, x2, y2 = d["box"]
                print(f"    {d['yolo_label']:<30} conf={d['confidence']:<6} "
                      f"box=({x1},{y1},{x2},{y2})")
        print("  Kept:")
        for d in after_nms:
            x1, y1, x2, y2 = d["box"]
            print(f"    {d['yolo_label']:<30} conf={d['confidence']:<6} "
                  f"box=({x1},{y1},{x2},{y2})")

    after_container = drop_container_boxes(after_nms, img_area)
    container_removed = [d for d in after_nms if d not in after_container]

    if debug:
        print(f"\n{'='*60}")
        print(f"STAGE 3 — after container suppression: "
              f"{len(after_container)} kept, {len(container_removed)} removed")
        print(f"{'='*60}")
        if container_removed:
            print("  Removed (large box containing >=2 smaller items):")
            for d in container_removed:
                x1, y1, x2, y2 = d["box"]
                area_pct = ((x2-x1)*(y2-y1)) / img_area * 100
                print(f"    {d['yolo_label']:<30} conf={d['confidence']:<6} "
                      f"covers {area_pct:.1f}% of image")
        print("  Kept (final):")
        for d in after_container:
            x1, y1, x2, y2 = d["box"]
            print(f"    {d['yolo_label']:<30} conf={d['confidence']:<6} "
                  f"box=({x1},{y1},{x2},{y2})")
        print()

    dets = sorted(after_container, key=lambda d: -d["confidence"])
    for det in dets:
        det["crop"] = img.crop(det["box"])
    return dets


def show_preview(image_path, dets,
                 window="Food detection -- press any key to close"):
    """Draw the detected regions on the photo and show it in an OpenCV window."""
    img = cv2.imread(str(image_path))
    if img is None:
        print("(could not open image for preview)")
        return

    palette = [(0, 0, 255), (0, 200, 0), (255, 100, 0),
               (200, 0, 200), (0, 200, 200), (255, 0, 128)]

    for i, det in enumerate(dets):
        x1, y1, x2, y2 = det["box"]
        color = palette[i % len(palette)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        text = f"Region {i}  conf={det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1, th + 6)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(img, text, (x1 + 3, ty - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    h, w = img.shape[:2]
    init_w = int(800 * w / max(h, w))
    init_h = int(800 * h / max(h, w))
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, init_w, init_h)
    cv2.imshow(window, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to a plate photo")
    ap.add_argument("--conf", type=float, default=0.1,
                    help="Minimum detection confidence (default 0.1)")
    ap.add_argument("--weights", default="yolov8m-oiv7.pt",
                    help="YOLO weights file (default yolov8m-oiv7.pt; "
                         "yolov8n-oiv7.pt is faster but lower recall)")
    ap.add_argument("--iou", type=float, default=0.45,
                    help="YOLO NMS IoU threshold (default 0.45; lower keeps "
                         "fewer overlapping boxes)")
    ap.add_argument("--max-det", type=int, default=50,
                    help="Max detections YOLO returns per image (default 50)")
    ap.add_argument("--agnostic-nms", action="store_true",
                    help="Class-agnostic NMS inside YOLO (merges boxes across "
                         "classes before they reach our post-processing)")
    ap.add_argument("--min-area", type=float, default=0.02,
                    help="Discard boxes smaller than this fraction of the image "
                         "area (default 0.02 = 2%%; raise to cut more clutter)")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="Print every raw YOLO detection and show what each "
                         "filtering stage keeps/removes")
    args = ap.parse_args()

    dets = detect_food(args.image, args.conf, args.weights,
                       args.iou, args.max_det, args.agnostic_nms,
                       args.min_area, args.debug)

    if not dets:
        print("No food regions detected. Try --conf 0.05, or check --min-area.")
        return

    print(f"Found {len(dets)} food region(s):\n")
    for i, det in enumerate(dets):
        x1, y1, x2, y2 = det["box"]
        print(f"  [{i}] conf={det['confidence']:<6} "
              f"box=({x1},{y1},{x2},{y2})  size={x2-x1}x{y2-y1}"
              f"  (yolo guess: \"{det['yolo_label']}\" — ignore, classifier will name this)")
    print("\nEach region above becomes one crop for the classifier to name.")

    if not args.no_preview:
        show_preview(args.image, dets)


if __name__ == "__main__":
    main()
