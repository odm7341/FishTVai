package com.fishtvai.usecase

import android.media.ImageProxy
import android.util.Log
import com.fishtvai.ml.InferenceEngine
import com.fishtvai.model.DisplayModel
import com.fishtvai.ml.util.ImageUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer

class MLProcessingUseCase(private val inferenceEngine: InferenceEngine) {

    private var initialized = false

    private suspend fun ensureInitialized() {
        if (!initialized) {
            inferenceEngine.initialize()
            initialized = true
        }
    }

    suspend fun processFrame(imageProxy: ImageProxy): DisplayModel =
        withContext(Dispatchers.IO) {
            try {
                ensureInitialized()

                val image = imageProxy.image
                    ?: throw IllegalStateException("Null image from proxy")

                val preprocessedBuffer: ByteBuffer = ImageUtils.processImageToTensor(image, 224, 224)
                    ?: throw IllegalStateException("Image preprocessing failed")

                inferenceEngine.runInference(preprocessedBuffer)

            } catch (e: IllegalArgumentException) {
                Log.e("MLPipeline", "Input Data Error: ${e.message}")
                throw IllegalStateException("Input data failure", e)
            } catch (e: IllegalStateException) {
                Log.e("MLPipeline", "System Failure during ML inference: ${e.message}")
                throw IllegalStateException("System failure during processing", e)
            } catch (e: Exception) {
                Log.e("MLPipeline", "Critical unhandled error in ML pipeline", e)
                throw RuntimeException("An unexpected error occurred during ML processing", e)
            } finally {
                imageProxy.close()
            }
        }

    fun release() {
        inferenceEngine.release()
        initialized = false
    }
}
