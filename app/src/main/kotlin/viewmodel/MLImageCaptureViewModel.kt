package com.fishtvai.ml.viewmodel

import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.fishtvai.ml.model.DisplayModel
import com.fishtvai.ml.util.InferenceEngine
import com.fishtvai.ml.util.MLCameraAnalyzer
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MLImageCaptureViewModel : ViewModel() {

    private val TAG = "MLImageCaptureVM"

    private val _analysisResult = MutableLiveData<DisplayModel?>()
    val analysisResult: LiveData<DisplayModel?> = _analysisResult

    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private lateinit var inferenceEngine: InferenceEngine
    private lateinit var analyzer: MLCameraAnalyzer

    fun initialize(modelPath: String) {
        inferenceEngine = InferenceEngine()
        try {
            inferenceEngine.initialize(modelPath)
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to initialize ML Inference Engine", e)
        }

        analyzer = MLCameraAnalyzer()
        analyzer.initialize(inferenceEngine, viewModelScope)
        android.util.Log.i(TAG, "ML Inference ViewModel initialized successfully.")
    }

    fun startAnalysis(cameraProvider: ProcessCameraProvider, cameraSelector: CameraSelector) {
        try {
            val imageAnalysis = ImageAnalysis.Builder()
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor, analyzer)
            android.util.Log.i(TAG, "Camera Analysis UseCase bound successfully.")
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Error binding ImageAnalysis UseCase.", e)
        }
    }

    fun cleanup() {
        android.util.Log.i(TAG, "Starting ML pipeline cleanup...")
        if (::inferenceEngine.isInitialized) {
            inferenceEngine.release()
            android.util.Log.i(TAG, "Inference Engine resources released.")
        }
        cameraExecutor.shutdown()
    }

    override fun onCleared() {
        super.onCleared()
        cleanup()
    }
}
