package com.fishtvai

import android.app.Application
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.fishtvai.usecase.MLProcessingUseCase
import com.fishtvai.viewmodel.MainViewModel

class MainViewModelFactory(
    private val application: Application,
    private val useCase: MLProcessingUseCase
) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
            return MainViewModel(application, useCase) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
