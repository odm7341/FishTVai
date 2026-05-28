package com.fishtvai

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.fishtvai.databinding.ActivityMainBinding
import com.fishtvai.ml.InferenceEngine
import com.fishtvai.model.DisplayModel
import com.fishtvai.usecase.MLProcessingUseCase
import com.fishtvai.viewmodel.MainViewModel
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var viewModel: MainViewModel
    private var isDetecting = false
    private var cameraProvider: ProcessCameraProvider? = null
    private var frameWidth = 0
    private var frameHeight = 0

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        val inferenceEngine = InferenceEngine(applicationContext, "fishtv.onnx")
        val useCase = MLProcessingUseCase(inferenceEngine)
        val factory = MainViewModelFactory(application, useCase)
        viewModel = ViewModelProvider(this, factory)[MainViewModel::class.java]

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.displayState.collect { displayModel ->
                    updateUI(displayModel)
                }
            }
        }

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.errorState.collect { failure ->
                    failure?.let {
                        Log.e("MainActivity", "Error: ${it.message}")
                        binding.detectionText.text = "Error: ${it.message}"
                    }
                }
            }
        }

        binding.toggleButton.setOnClickListener {
            if (isDetecting) stopDetection() else startDetection()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return
        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        val preview = Preview.Builder()
            .setTargetRotation(binding.previewView.display.rotation)
            .build()
        preview.setSurfaceProvider(binding.previewView.surfaceProvider)

        val imageAnalysis = ImageAnalysis.Builder()
            .build()

        imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
            frameWidth = imageProxy.width
            frameHeight = imageProxy.height
            val image = imageProxy.image
            if (image != null && isDetecting) {
                val tensor = com.fishtvai.ml.util.ImageUtils.processImageToTensor(image, 640, 640)
                imageProxy.close()
                if (tensor != null) {
                    viewModel.processFrame(tensor, frameWidth, frameHeight)
                }
            } else {
                imageProxy.close()
            }
        }

        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            this,
            cameraSelector,
            imageAnalysis,
            preview
        )
    }

    private fun startDetection() {
        isDetecting = true
        binding.toggleButton.text = getString(R.string.stop_detection)
        binding.overlayCard.visibility = View.VISIBLE
        Log.d("MainActivity", "Detection started")
    }

    private fun stopDetection() {
        isDetecting = false
        binding.toggleButton.text = getString(R.string.start_detection)
        binding.overlayCard.visibility = View.GONE
        binding.detectionText.text = getString(R.string.no_detections)
        binding.boundingBoxOverlay.setDetections(emptyList(), 1, 1)
        Log.d("MainActivity", "Detection stopped")
    }

    private var lastFrameTime = 0L
    private fun updateUI(displayModel: DisplayModel) {
        val now = System.currentTimeMillis()
        val dt = now - lastFrameTime
        lastFrameTime = now

        Log.d("MainActivity", "updateUI: ${displayModel.detections.size} detections, dt=${dt}ms")

        binding.detectionText.text = if (displayModel.detections.isEmpty()) {
            getString(R.string.no_detections)
        } else {
            displayModel.detections.joinToString("\n") { detection ->
                "${detection.label}: ${"%.1f".format(detection.confidence * 100)}%"
            }
        }
        binding.fpsText.text = "${displayModel.detections.size} objects, ${dt}ms"

        if (frameWidth > 0 && frameHeight > 0) {
            binding.boundingBoxOverlay.setDetections(displayModel.detections, frameWidth, frameHeight)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
