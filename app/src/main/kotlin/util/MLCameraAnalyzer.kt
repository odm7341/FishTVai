package com.fishtvai.ml.util

import android.util.Log
import androidx.annotation.NonNull
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.fishtvai.ml.model.DisplayModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class MLCameraAnalyzer : ImageAnalysis.Analyzer {

    private val TAG = "MLCameraAnalyzer"

    private var inferenceEngine: InferenceEngine? = null
    private var scope: CoroutineScope? = null

    fun initialize(engine: InferenceEngine, scope: CoroutineScope?) {
        this.inferenceEngine = engine
        this.scope = scope
    }

    @NonNull
    override fun analyze(imageProxy: ImageProxy) {
        val image = imageProxy.image
        if (image == null) {
            Log.w(TAG, "Received null Image object from ImageProxy.")
            imageProxy.close()
            return
        }

        scope?.launch(Dispatchers.IO) {
            try {
                Log.d(TAG, "Starting image processing and tensor conversion.")

                val inputTensor = ImageUtils.processImageToTensor(image, 640, 640)

                if (inputTensor == null) {
                    Log.e(TAG, "Failed to convert ImageProxy to Tensor.")
                    return@launch
                }

                Log.d(TAG, "Starting ONNX Inference execution.")

                val result = inferenceEngine?.runInference(inputTensor)

                result?.let {
                    Log.i(TAG, "Inference successful. Detected Class: ${it.predictedClass} with confidence: ${"%.2f".format(it.confidence)}%")
                }

            } catch (e: Exception) {
                Log.e(TAG, "CRITICAL: Error during ML pipeline execution: ${e.message}", e)
            } finally {
                imageProxy.close()
            }
        }
    }
}
