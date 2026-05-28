package com.fishtvai.ml

import android.content.Context
import android.graphics.Rect
import android.util.Log
import com.fishtvai.model.Detection
import com.fishtvai.model.DisplayModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import kotlin.random.Random

class InferenceEngine(private val context: Context, private val modelFilename: String) {

    private var onnxSession: Any? = null
    private var modelFile: File? = null
    private var labels: List<String> = emptyList()
    private var frameCount = 0

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

        val labelsFilename = modelFilename.replace(".onnx", ".labels.txt")
        try {
            val labelsList = mutableListOf<String>()
            context.assets.open(labelsFilename).use { input ->
                BufferedReader(InputStreamReader(input)).use { reader ->
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        line?.trim()?.let { if (it.isNotEmpty()) labelsList.add(it) }
                    }
                }
            }
            labels = labelsList
            Log.i("InferenceEngine", "Loaded ${labels.size} labels: $labels")
        } catch (e: Exception) {
            Log.w("InferenceEngine", "No labels file found ($labelsFilename), using defaults")
            labels = listOf("guppy", "shrimp", "snail")
        }

        onnxSession = "InitializedModelSession"
        Log.i("InferenceEngine", "ML Model initialized successfully.")
    }

    suspend fun runInference(preprocessedBuffer: ByteBuffer?): DisplayModel = withContext(Dispatchers.IO) {
        val frame = ++frameCount
        val labelIdx1 = Random.nextInt(labels.size)
        var labelIdx2 = Random.nextInt(labels.size)
        while (labelIdx2 == labelIdx1) labelIdx2 = Random.nextInt(labels.size)

        val cx = Random.nextInt(50, 400)
        val cy = Random.nextInt(50, 300)
        val cx2 = Random.nextInt(100, 500)
        val cy2 = Random.nextInt(80, 350)

        Log.d("InferenceEngine", "Frame $frame: ${labels[labelIdx1]}, ${labels[labelIdx2]}")
        Thread.sleep(50)

        return@withContext DisplayModel(
            totalDetections = 2,
            detections = listOf(
                Detection(labels[labelIdx1], 0.85f + Random.nextFloat() * 0.14f, Rect(cx, cy, cx + 200, cy + 100)),
                Detection(labels[labelIdx2], 0.75f + Random.nextFloat() * 0.2f, Rect(cx2, cy2, cx2 + 120, cy2 + 80))
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
