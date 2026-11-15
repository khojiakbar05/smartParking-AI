# Per-slot classification engine (Phase 2)

## What is implemented

- `slot_crops.py` keeps slot IDs and makes perspective rectifications for convex
  quadrilaterals. Other polygons use a masked bounding crop. It validates the
  input frame and rejects coordinates outside it.
- `slot_classifier.py` defines a replaceable classifier interface and the
  `OCCUPIED` / `VACANT` / `UNKNOWN` contract. Until a trained checkpoint is
  supplied, the live classifier is not enabled. Low confidence stays UNKNOWN.
  Vehicle overlap may support an UNKNOWN result, but does not override a
  confident slot-classifier result.
- `parking_detection.py --slot-classifier PATH` opts into the new batch
  classifier, temporal smoothing, green/red/yellow polygons, slot confidence,
  counts, and independent blue YOLO boxes. Without that option the existing
  detection path is unchanged.
- `collect_slot_samples.py` lets a person review a live slot crop and label it
  using `O` or `V`. No YOLO result is used to create a label. Exact duplicate
  crops are rejected, conflicting labels are reported, writes are validated
  and atomic, and the JSONL manifest records manual provenance.
- `train_slot_classifier.py` is separate from live inference. It supports
  MobileNetV3-Small (default) and EfficientNet-B0; pretrained ImageNet weights
  are opt-in (`--pretrained`) and no training or download has been run.

## Crop validation performed

Five saved frames captured from the selected iPhone camera were used as source
frames. The diagnostics generated each of the current 7-slot crops and each of
the historical 16-slot crops from the old Git snapshot without editing either
slot coordinate set. Perspective output preserved slot IDs and all crops
decoded successfully. Sample views are in:

- `debug/slot_crop_validation_7/frame_05_contact.jpg`
- `debug/slot_crop_validation_7/frame_05_polygons.jpg`
- `debug/slot_crop_validation_16/frame_05_contact.jpg`
- `debug/slot_crop_validation_16/frame_05_polygons.jpg`

The current 7-slot polygons line up with the visible bays in the saved frame.
The historical 16-slot geometry was also crop-tested, but its polygons do not
consistently cover full bays in this same view. It appears to belong to a
different scene or camera framing; validate that configuration against its
original view before using it for labels or occupancy. This is a geometry
review only, not an occupancy accuracy result. The saved view contains a
parking-lot image on a display, rather than a direct view of a physical lot.

The pre-existing loader also warns that historical P03 has crossing vertices;
it repairs ordering in memory for this run and leaves JSON untouched.

## Model and dataset assessment

Start with MobileNetV3-Small using ImageNet transfer learning, then fine-tune
with manually reviewed samples from the target iPhone, camera mount, lighting,
and slot coordinates. Its small architecture is a sensible real-time baseline;
EfficientNet-B0 is included as a heavier comparison candidate. Both model
constructors were instantiated locally without pretrained weights or network
downloads. No target-domain validation set exists yet, so neither model has
measured occupancy accuracy.

PKLot provides individually cropped parking-space occupancy images across
three lots/views and is a reasonable initial warm-up dataset
([dataset paper](https://www.sciencedirect.com/science/article/pii/S0957417415001086),
[maintainer page](https://web.inf.ufpr.br/luizoliveira/research-interests/pklot/)).
CNRPark+EXT provides labeled vacant/occupied parking-space patches
([dataset site](https://cnrpark.it/)). Both datasets can help initialize a
slot classifier, but camera angle, crop geometry, weather, display artifacts,
and local parking appearance differ from this camera. Fine-tuning and
validation on manually labeled target-camera samples are required. The code
does not download either dataset.

## Commands

Run crop previews against the five saved camera frames (from repository root):

```bash
ai-service/.venv311/bin/python ai-service/slot_crop_diagnostics.py \
  --input-dir ai-service/debug/camera_20261116_035222 \
  --output-dir ai-service/debug/slot_crop_validation_7

ai-service/.venv311/bin/python ai-service/slot_crop_diagnostics.py \
  --input-dir ai-service/debug/camera_20261116_035222 \
  --slots ai-service/debug/slots_16_historical.json \
  --output-dir ai-service/debug/slot_crop_validation_16
```

Collect actual camera samples (review the highlighted crop before pressing a
label key):

```bash
cd ai-service
.venv311/bin/python collect_slot_samples.py
```

Keyboard: `N` / `P` selects the next / previous slot, `O` saves OCCUPIED, `V`
saves VACANT, `Q` quits. Collection requires a person to see the camera view
and approve each label.

Train only after collecting at least five crops per label. Pretrained weights
are not fetched unless `--pretrained` is explicitly added:

```bash
cd ai-service
.venv311/bin/python train_slot_classifier.py \
  --data-dir dataset/slot_samples \
  --output models/slot_classifier.pt \
  --architecture mobilenet_v3_small
```

Launch with an already trained checkpoint:

```bash
cd ai-service
.venv311/bin/python parking_detection.py \
  --slot-classifier models/slot_classifier.pt
```

Without a trained checkpoint, the existing application command remains:

```bash
cd ai-service
.venv311/bin/python parking_detection.py
```

## Verification and remaining work

`python -m unittest discover -s ai-service -p 'test_*.py' -v` passes all 22
tests, including crop ID/mask/bounds checks, UNKNOWN behavior, fusion, and
smoothing. Python compilation passed for all new modules and the modified
application. MobileNetV3-Small and EfficientNet-B0 both instantiated without
downloads. A temporary random MobileNetV3-Small checkpoint also completed CPU
inference and returned UNKNOWN at 0.5052 confidence; the temporary checkpoint
was removed. Live camera collection was not run during this implementation;
the crop pipeline was checked using the five already captured iPhone frames.
There are no manually labeled samples or trained slot-classifier checkpoint
yet, so meaningful per-slot occupancy results and accuracy remain unverified.
