package com.fishtvai.ml

import android.content.Context
import android.util.Log
import com.fishtvai.model.Detection
import com.fishtvai.model.DisplayModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer

class InferenceEngine(private val context: Context, private val modelFilename: String) {

    private var onnxSession: Any? = null
    private var modelFile: File? = null

    suspend fun initialize() = withContext(Dispatchers.IO) {
        Log.d("InferenceEngine", "Starting model initialization...")

        val targetFile = File(context.filesDir, modelFilename)
        modelFile = targetFile

        if (!targetFile.exists()) {
            Log.i("InferenceEngine", "Copying model from assets to $targetFile")
            context.assets.open(modelFilename).use { input ->
                FileOutputStream(targetFile).use { output ->
                    input.copyTo(output)
                }
            }
            Log.i("InferenceEngine", "Model copied (${targetFile.length()} bytes)")
        }

        onnxSession = "InitializedModelSession"
        Log.i("InferenceEngine", "ML Model initialized successfully.")
    }

    suspend fun runInference(preprocessedBuffer: ByteBuffer): DisplayModel = withContext(Dispatchers.IO) {
        Log.d("InferenceEngine", "Running inference with preprocessed buffer... (Simulation)")
        Thread.sleep(100)

        return@withContext DisplayModel(
            totalDetections = 2,
            detections = listOf(
                Detection("Fish", 0.95f, android.graphics.Rect(100, 50, 300, 150)),
                Detection("Wave", 0.88f, android.graphics.Rect(50, 200, 500, 400))
            )
        )
    }

    fun release() {
        onnxSession = null
        Log.d("InferenceEngine", "ML Model resources released.")
    }

    suspend fun <T> executeWithResourceGuard(block: suspend (InferenceEngine) -> T): T {
        try {
            if (onnxSession == null) throw IllegalStateException("Engine must be initialized first.")
            return block(this)
        } finally {
            this.release()
        }
    }
}
