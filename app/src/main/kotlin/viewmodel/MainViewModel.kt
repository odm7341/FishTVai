package com.fishtvai.viewmodel

import android.app.Application
import androidx.camera.core.ImageProxy
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.fishtvai.model.DisplayModel
import com.fishtvai.model.FailureReason
import com.fishtvai.usecase.MLProcessingUseCase
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(application: Application, private val useCase: MLProcessingUseCase) : AndroidViewModel(application) {

    private val _displayState = MutableStateFlow<DisplayModel>(DisplayModel())
    val displayState: StateFlow<DisplayModel> = _displayState

    private val _errorState = MutableStateFlow<FailureReason?>(null)
    val errorState: StateFlow<FailureReason?> = _errorState

    fun processImageFrame(imageProxy: ImageProxy) {
        viewModelScope.launch {
            try {
                val result = useCase.processFrame(imageProxy)
                _displayState.update { result }
                _errorState.value = null
            } catch (e: Exception) {
                val failure = FailureReason(
                    code = e::class.simpleName?.uppercase() ?: "UNKNOWN",
                    message = e.message ?: "Unknown error during processing",
                    isCritical = e is IllegalStateException
                )
                _errorState.update { failure }
                Log.e("MainViewModel", "Processing failed", e)
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        useCase.release()
    }
}
