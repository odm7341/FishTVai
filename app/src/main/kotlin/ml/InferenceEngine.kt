package com.fishtvai.ml

import android.content.Context
import android.graphics.Rect
import android.util.Log
import com.fishtvai.model.Detection
import com.fishtvai.model.DisplayModel
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.FloatBuffer

class InferenceEngine(private val context: Context, private val modelFilename: String) {

    private var environment: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var labels: List<String> = emptyList()
    private var inputName: String = ""
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
            Log.w("InferenceEngine", "No labels file found, using defaults")
            labels = listOf("guppy", "shrimp", "snail")
        }

        try {
            environment = OrtEnvironment.getEnvironment()

            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setIntraOpNumThreads(2)
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

            session = environment!!.createSession(targetFile.absolutePath, sessionOptions)

            val inputInfo = session!!.inputInfo
            inputName = inputInfo.keys.first()
            Log.i("InferenceEngine", "Model input: '$inputName', ${inputInfo.values.first()}")

            val outputInfo = session!!.outputInfo
            Log.i("InferenceEngine", "Model outputs: ${outputInfo.keys}")

            Log.i("InferenceEngine", "ONNX Runtime initialized successfully.")
        } catch (e: Exception) {
            Log.e("InferenceEngine", "Failed to initialize ONNX Runtime", e)
            Log.w("InferenceEngine", "Falling back to simulated inference")
            environment = null
            session = null
        }
    }

    suspend fun runInference(tensorBuffer: ByteBuffer): DisplayModel = withContext(Dispatchers.IO) {
        val env = environment
        val sess = session

        if (env != null && sess != null) {
            runRealInference(env, sess, tensorBuffer)
        } else {
            runSimulatedInference()
        }
    }

    private fun runRealInference(env: OrtEnvironment, sess: OrtSession, tensorBuffer: ByteBuffer): DisplayModel {
        try {
            tensorBuffer.rewind()
            val floatBuffer = tensorBuffer.asFloatBuffer()

            val inputShape = longArrayOf(1, 3, 224, 224)
            val inputTensor = OnnxTensor.createTensor(env, floatBuffer, inputShape)

            val inputs = mapOf(inputName to inputTensor)
            val results = sess.run(inputs)

            val outputTensor = results[0] as? OnnxTensor
                ?: throw RuntimeException("Unexpected output type")

            val outputBuffer = outputTensor.floatBuffer
            val outputArray = FloatArray(outputBuffer.remaining())
            outputBuffer.get(outputArray)

            inputTensor.close()
            results.close()

            val detections = parseOutput(outputArray)
            return DisplayModel(
                totalDetections = detections.size,
                detections = detections
            )
        } catch (e: Exception) {
            Log.e("InferenceEngine", "ONNX inference failed", e)
            throw RuntimeException("ONNX inference failed", e)
        }
    }

    private fun parseOutput(outputArray: FloatArray): List<Detection> {
        if (outputArray.isEmpty()) return emptyList()

        if (outputArray.size == labels.size) {
            val maxIndex = outputArray.indices.maxByOrNull { outputArray[it] } ?: return emptyList()
            val confidence = outputArray[maxIndex]
            val label = labels.getOrElse(maxIndex) { "class_$maxIndex" }
            if (confidence > 0.3f) {
                return listOf(
                    Detection(label, confidence, Rect(100, 80, 400, 300))
                )
            }
            return emptyList()
        }

        val detections = mutableListOf<Detection>()
        val numClasses = labels.size
        val numDetections = outputArray.size / (numClasses + 4)

        for (i in 0 until numDetections.coerceAtMost(10)) {
            val offset = i * (numClasses + 4)
            val cx = outputArray[offset] * 640
            val cy = outputArray[offset + 1] * 480
            val w = outputArray[offset + 2] * 640
            val h = outputArray[offset + 3] * 480

            val scores = outputArray.sliceArray(offset + 4 until offset + 4 + numClasses)
            val maxIndex = scores.indices.maxByOrNull { scores[it] } ?: continue
            val confidence = scores[maxIndex]

            if (confidence > 0.3f) {
                val x = (cx - w / 2).toInt()
                val y = (cy - h / 2).toInt()
                val label = labels.getOrElse(maxIndex) { "class_$maxIndex" }
                detections.add(
                    Detection(label, confidence, Rect(x, y, x + w.toInt(), y + h.toInt()))
                )
            }
        }

        return detections
    }

    private fun runSimulatedInference(): DisplayModel {
        Log.d("InferenceEngine", "Running simulated inference")
        Thread.sleep(30)
        return DisplayModel(
            totalDetections = 3,
            detections = listOf(
                Detection(labels.getOrElse(0) { "guppy" }, 0.97f, Rect(180, 120, 340, 210)),
                Detection(labels.getOrElse(1) { "shrimp" }, 0.84f, Rect(400, 280, 490, 340)),
                Detection(labels.getOrElse(2) { "snail" }, 0.72f, Rect(60, 330, 150, 400))
            )
        )
    }

    fun release() {
        try {
            session?.close()
            Log.i("InferenceEngine", "ONNX session closed.")
        } catch (e: Exception) {
            Log.w("InferenceEngine", "Error closing session: ${e.message}")
        }
        session = null
        environment = null
    }

    suspend fun <T> executeWithResourceGuard(block: suspend (InferenceEngine) -> T): T {
        try {
            if (session == null) throw IllegalStateException("Engine must be initialized first.")
            return block(this)
        } finally {
            this.release()
        }
    }
}
