package com.fishtvai.ml.model

class DisplayModel {
    var confidence: Float = 0f
    var predictedClass: String = ""
    var rawOutputVector: FloatArray = floatArrayOf()
        get() = field.clone()
}
