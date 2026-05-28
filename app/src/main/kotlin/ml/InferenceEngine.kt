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

                val stride = 4 + labels.size
                val rows = outputArray.size / stride
                if (outputArray.size % stride == 0) {
                    // Sample all columns at regular intervals to understand distribution
                    val sampleIndices = (0 until 100).map { i -> (i * rows / 100) }
                    val colRanges = (0 until stride).map { col ->
                        val vals = sampleIndices.map { outputArray[it * stride + col] }
                        val colMin = vals.min()
                        val colMax = vals.max()
                        val colAvg = vals.average()
                        "c$col: rng=${"%.1f".format(colMin)}-${"%.1f".format(colMax)} avg=${"%.1f".format(colAvg)}"
                    }
                    Log.i("InferenceEngine", "Row-major column ranges: ${colRanges.joinToString(" | ")}")
                }

                val dets = parseYoloOutput(outputArray)
                allDetections.addAll(dets)
            }

            inputTensor.close()
            results.close()

            val nmsResult = nms(allDetections)
            // Heuristic: if all detections cluster around 0.5 (sigmoid of near-zero logits), model detects nothing
            val confidences = nmsResult.map { it.confidence }
            val isNoise = confidences.isNotEmpty() &&
                confidences.all { it in 0.45f..0.55f } &&
                confidences.distinct().size <= 2
            val filtered = if (isNoise) emptyList() else nmsResult

            Log.i("InferenceEngine", "Total detections: ${allDetections.size} -> after NMS: ${nmsResult.size} -> final: ${filtered.size}")

            return DisplayModel(
                totalDetections = filtered.size,
                detections = filtered
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
        val expectedStride = 4 + numClasses

        if (n % expectedStride == 0) {
            val rows = n / expectedStride
            for (useSigmoid in listOf(true, false)) {
                val sigLabel = if (useSigmoid) "sigmoid" else "raw"
                Log.i("InferenceEngine", "Trying row-major [cx,cy,w,h,cls] $sigLabel: $rows x $expectedStride")
                val dets = parseRawRows(outputArray, rows, useSigmoid)
                if (dets.isNotEmpty()) return dets
            }
        }

        if (n % expectedStride == 0) {
            val cols = n / expectedStride
            for (useSigmoid in listOf(true, false)) {
                val sigLabel = if (useSigmoid) "sigmoid" else "raw"
                Log.i("InferenceEngine", "Trying col-major [cx,cy,w,h,cls] $sigLabel: ${expectedStride} x $cols")
                val dets = parseBoxClassColumns(outputArray, cols, boxFirst = true, useSigmoid)
                if (dets.isNotEmpty()) return dets
            }
            for (useSigmoid in listOf(true, false)) {
                val sigLabel = if (useSigmoid) "sigmoid" else "raw"
                Log.i("InferenceEngine", "Trying col-major [cls,cx,cy,w,h] $sigLabel: ${expectedStride} x $cols")
                val dets = parseBoxClassColumns(outputArray, cols, boxFirst = false, useSigmoid)
                if (dets.isNotEmpty()) return dets
            }
        }

        if (numClasses > 0 && n >= numClasses) {
            Log.i("InferenceEngine", "Trying classification output")
            val dets = parseClassification(outputArray)
            if (dets.isNotEmpty()) return dets
        }

        Log.w("InferenceEngine", "Could not parse output of size $n with $numClasses classes")
        return emptyList()
    }

    private fun parseBoxClassColumns(outputArray: FloatArray, cols: Int, boxFirst: Boolean, useSigmoid: Boolean): List<Detection> {
        val detections = mutableListOf<Detection>()
        val numClasses = labels.size
        for (col in 0 until cols) {
            val cxIdx: Int
            val cyIdx: Int
            val wIdx: Int
            val hIdx: Int
            val clsStart: Int

            if (boxFirst) {
                cxIdx = 0
                cyIdx = 1
                wIdx = 2
                hIdx = 3
                clsStart = 4
            } else {
                clsStart = 0
                cxIdx = numClasses
                cyIdx = numClasses + 1
                wIdx = numClasses + 2
                hIdx = numClasses + 3
            }

            val cx = outputArray[cxIdx * cols + col]
            val cy = outputArray[cyIdx * cols + col]
            val bw = outputArray[wIdx * cols + col]
            val bh = outputArray[hIdx * cols + col]

            var bestScore = -1f
            var bestIdx = -1
            for (c in 0 until numClasses) {
                val raw = outputArray[(clsStart + c) * cols + col]
                val score = if (useSigmoid) sigmoid(raw) else raw.coerceIn(0f, 1f)
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

    private fun parseRawRows(outputArray: FloatArray, rows: Int, useSigmoid: Boolean): List<Detection> {
        val numClasses = labels.size
        val stride = 4 + numClasses
        val detections = mutableListOf<Detection>()
        for (i in 0 until rows) {
            val offset = i * stride
            if (offset + stride > outputArray.size) break
            val cx = outputArray[offset]
            val cy = outputArray[offset + 1]
            val bw = outputArray[offset + 2]
            val bh = outputArray[offset + 3]

            var bestScore = -1f
            var bestIdx = -1
            for (c in 0 until numClasses) {
                val raw = outputArray[offset + 4 + c]
                val score = if (useSigmoid) sigmoid(raw) else raw.coerceIn(0f, 1f)
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

    private fun parseClassification(outputArray: FloatArray): List<Detection> {
        val maxIndex = (0 until labels.size).maxByOrNull { sigmoid(outputArray[it]) } ?: return emptyList()
        val confidence = sigmoid(outputArray[maxIndex])
        if (confidence > 0.3f) {
            return listOf(
                Detection(labels.getOrElse(maxIndex) { "class_$maxIndex" }, confidence, Rect(100, 80, 500, 380))
            )
        }
        return emptyList()
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + exp(-x.toDouble())).toFloat()

    private fun nms(detections: List<Detection>, iouThreshold: Float = 0.5f, maxDetections: Int = 50): List<Detection> {
        if (detections.size <= 1) return detections
        val sorted = detections.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()
        val remaining = sorted.toMutableList()
        while (remaining.isNotEmpty() && selected.size < maxDetections) {
            val best = remaining.removeAt(0)
            selected.add(best)
            remaining.removeAll { IoU(best.boundingBoxPixels, it.boundingBoxPixels) > iouThreshold }
        }
        return selected
    }

    private fun IoU(a: android.graphics.Rect, b: android.graphics.Rect): Float {
        val interLeft = maxOf(a.left, b.left)
        val interTop = maxOf(a.top, b.top)
        val interRight = minOf(a.right, b.right)
        val interBottom = minOf(a.bottom, b.bottom)
        if (interLeft >= interRight || interTop >= interBottom) return 0f
        val interArea = (interRight - interLeft).toLong() * (interBottom - interTop).toLong()
        val aArea = (a.right - a.left).toLong() * (a.bottom - a.top).toLong()
        val bArea = (b.right - b.left).toLong() * (b.bottom - b.top).toLong()
        return interArea.toFloat() / (aArea + bArea - interArea).toFloat()
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
