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
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

class InferenceEngine(private val context: Context, private val modelFilename: String) {

    private var environment: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var labels: List<String> = emptyList()
    private var inputName: String = ""
    private var outputNames: List<String> = emptyList()
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
            Log.i("InferenceEngine", "Model input: '$inputName'")

            val outputs = session!!.outputInfo
            outputNames = outputs.keys.toList()
            Log.i("InferenceEngine", "Model outputs: $outputNames")
            for ((name, _) in outputs) {
                Log.i("InferenceEngine", "  Output: '$name'")
            }

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
            val inputShape = longArrayOf(1, 3, 640, 640)
            val inputTensor = OnnxTensor.createTensor(env, floatBuffer, inputShape)
            val inputs = mapOf(inputName to inputTensor)
            val results = sess.run(inputs)

            val allDetections = mutableListOf<Detection>()

            for (outputIdx in 0 until results.size()) {
                val outputTensor = results[outputIdx] as? OnnxTensor ?: continue
                val outputBuffer = outputTensor.floatBuffer
                val outputArray = FloatArray(outputBuffer.remaining())
                outputBuffer.get(outputArray)

                val outName = if (outputIdx < outputNames.size) outputNames[outputIdx] else "output_$outputIdx"
                Log.i("InferenceEngine", "Output '$outName': ${outputArray.size} floats, first 10: ${outputArray.take(10)}")

                val dets = parseYoloOutput(outputArray)
                allDetections.addAll(dets)
            }

            inputTensor.close()
            results.close()

            Log.i("InferenceEngine", "Total detections: ${allDetections.size}")
            for (d in allDetections) {
                Log.i("InferenceEngine", "  ${d.label} conf=${d.confidence} box=${d.boundingBoxPixels}")
            }

            return DisplayModel(
                totalDetections = allDetections.size,
                detections = allDetections
            )
        } catch (e: Exception) {
            Log.e("InferenceEngine", "ONNX inference failed", e)
            throw RuntimeException("ONNX inference failed", e)
        }
    }

    private fun parseYoloOutput(outputArray: FloatArray): List<Detection> {
        if (outputArray.isEmpty()) return emptyList()

        val numClasses = labels.size
        val n = outputArray.size

        if (n >= 6 * 5 && n % 6 == 0) {
            val cols = n / 6
            Log.i("InferenceEngine", "Trying NMS column format: 6 x $cols")
            val dets = parseBoxNmsColumns(outputArray, cols)
            if (dets.isNotEmpty()) return dets
        }

        val stride = 4 + numClasses
        if (stride > 4 && n % stride == 0) {
            val cols = n / stride
            Log.i("InferenceEngine", "Trying raw column-major: $stride x $cols")
            val dets = parseRawColumns(outputArray, cols, stride)
            if (dets.isNotEmpty()) return dets
        }

        if (stride > 4 && n % stride == 0) {
            val rows = n / stride
            Log.i("InferenceEngine", "Trying raw row-major: $rows x $stride")
            val dets = parseRawRows(outputArray, rows, stride)
            if (dets.isNotEmpty()) return dets
        }

        if (numClasses > 0 && n >= numClasses) {
            Log.i("InferenceEngine", "Trying classification output")
            val maxIndex = (0 until numClasses).maxByOrNull { outputArray[it] } ?: return emptyList()
            val confidence = sigmoid(outputArray[maxIndex])
            if (confidence > 0.3f) {
                return listOf(
                    Detection(labels.getOrElse(maxIndex) { "class_$maxIndex" }, confidence, Rect(100, 80, 500, 380))
                )
            }
        }

        Log.w("InferenceEngine", "Could not parse output of size $n with $numClasses classes")
        return emptyList()
    }

    private fun parseBoxNmsColumns(outputArray: FloatArray, cols: Int): List<Detection> {
        val detections = mutableListOf<Detection>()
        for (col in 0 until cols) {
            val x1 = outputArray[col]
            val y1 = outputArray[cols + col]
            val x2 = outputArray[2 * cols + col]
            val y2 = outputArray[3 * cols + col]
            val score = sigmoid(outputArray[4 * cols + col])
            val classId = outputArray[5 * cols + col].toInt()

            if (score > 0.3f && classId in labels.indices) {
                val label = labels[classId]
                detections.add(
                    Detection(
                        label, score,
                        Rect(x1.toInt(), y1.toInt(), x2.toInt(), y2.toInt())
                    )
                )
            }
        }
        return detections
    }

    private fun parseRawColumns(outputArray: FloatArray, cols: Int, stride: Int): List<Detection> {
        val detections = mutableListOf<Detection>()
        for (col in 0 until cols) {
            val cx = outputArray[col]
            val cy = outputArray[cols + col]
            val bw = outputArray[2 * cols + col]
            val bh = outputArray[3 * cols + col]

            var bestScore = -1f
            var bestIdx = -1
            for (c in 0 until labels.size) {
                val score = sigmoid(outputArray[(4 + c) * cols + col])
                if (score > bestScore) {
                    bestScore = score
                    bestIdx = c
                }
            }

            if (bestScore > 0.3f && bestIdx >= 0) {
                val x = (cx - bw / 2f).coerceIn(0f, 639f)
                val y = (cy - bh / 2f).coerceIn(0f, 639f)
                val w = bw.coerceIn(0f, 639f - x)
                val h = bh.coerceIn(0f, 639f - y)
                detections.add(
                    Detection(
                        labels.getOrElse(bestIdx) { "class_$bestIdx" },
                        bestScore,
                        Rect(x.toInt(), y.toInt(), (x + w).toInt(), (y + h).toInt())
                    )
                )
            }
        }
        return detections
    }

    private fun parseRawRows(outputArray: FloatArray, rows: Int, stride: Int): List<Detection> {
        val detections = mutableListOf<Detection>()
        for (i in 0 until rows) {
            val offset = i * stride
            if (offset + 4 + labels.size > outputArray.size) break

            val cx = outputArray[offset]
            val cy = outputArray[offset + 1]
            val bw = outputArray[offset + 2]
            val bh = outputArray[offset + 3]

            var bestScore = -1f
            var bestIdx = -1
            for (c in 0 until labels.size) {
                val score = sigmoid(outputArray[offset + 4 + c])
                if (score > bestScore) {
                    bestScore = score
                    bestIdx = c
                }
            }

            if (bestScore > 0.3f && bestIdx >= 0) {
                val x = (cx - bw / 2f).coerceIn(0f, 639f)
                val y = (cy - bh / 2f).coerceIn(0f, 639f)
                val w = bw.coerceIn(0f, 639f - x)
                val h = bh.coerceIn(0f, 639f - y)
                detections.add(
                    Detection(
                        labels.getOrElse(bestIdx) { "class_$bestIdx" },
                        bestScore,
                        Rect(x.toInt(), y.toInt(), (x + w).toInt(), (y + h).toInt())
                    )
                )
            }
        }
        return detections
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + exp(-x.toDouble())).toFloat()

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
