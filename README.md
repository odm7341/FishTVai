# FishTV AI

Real-time ONNX-based object detection for aquarium cameras. Captures 1920×1080 camera frames, preprocesses them to 640×640 (squish to square), and runs ONNX inference to detect guppies, shrimp, and snails.

Built with CameraX + ONNX Runtime for Android.

## Architecture

```
CameraX ImageAnalysis
       ↓
  ImageUtils.processImageToTensor()
  (YUV_420_888 → RGB → squish to 640×640 → CHW float tensor)
       ↓
  InferenceEngine.runInference()
  (ONNX Runtime → NMS → filtered detections)
       ↓
  MainViewModel → updateUI()
  (overlay boxes + detection text + FPS)
```

## Key Files

| File | Purpose |
|---|---|
| `MainActivity.kt` | Camera setup, UI updates, toggle detection |
| `ml/InferenceEngine.kt` | ONNX model loading + inference + output parsing |
| `util/ImageUtils.kt` | YUV→RGB→640×640→CHW float tensor + debug bitmap |
| `model/DisplayModel.kt` | `Detection` and `DisplayModel` data classes |
| `usecase/MLProcessingUseCase.kt` | Lazy init + inference orchestration |
| `viewmodel/MainViewModel.kt` | State management for detection results |
| `view/BoundingBoxOverlay.kt` | Custom View for drawing detection boxes |

## Model

- Input: `"images"`, float32, shape `[1, 3, 640, 640]`
- Output: `"output0"`, shape `[1, 7, 8400]` (4 box coords + 3 class probs)
- Classes: `guppy`, `shrimp`, `snail` (from `fishtv.labels.txt`)
- Model producer: PyTorch 2.11.0, opset 20. Class scores have internal Sigmoid.

## Building

```bash
git clone <repo>
cd FishTVai
./gradlew assembleDebug
```

Place `fishtv.onnx` and `fishtv.labels.txt` in `app/src/main/assets/` (copied to device storage on first run).

## Usage

- App is locked to **landscape orientation**
- Tap "Start Detection" to begin (runs inference every 1s)
- Preprocessed 640×640 image shown on the left, detection list on the right
- Bounding boxes overlay the preprocessed image
- Confidence threshold: 50%

## Next Steps

- **Video streaming**: Host an MJPEG or WebRTC stream from the camera preview so the aquarium can be viewed remotely.
- **Web dashboard**: Serve a webpage (e.g. via an embedded HTTP server or a companion process) showing real-time detection stats — histograms of detections over time, per-class counts, confidence trends.
- **Detection API**: Expose a REST or WebSocket API so external clients can query the latest detections, subscribe to detection events, or pull historical stats.
- **Persistence**: Store detection events in a local database (Room) with timestamps to power the stats dashboard.
- **Background service**: Run detection as a foreground service so it continues even when the UI is closed, keeping the video stream and API live.
