// DisplayModel.kt
package com.fishtvai.model

/**
 * Standardized data model for displaying detection results.
 * Represents a single identified object.
 */
data class Detection(
    val label: String,
    val confidence: Float,
    val boundingBoxPixels: android.graphics.Rect
)

/**
 * Represents the overall state of the ML detection results on the UI.
 */
data class DisplayModel(
    val timestamp: Long = System.currentTimeMillis(),
    val totalDetections: Int = 0,
    val detections: List<Detection> = emptyList()
)

/**
 * Used to communicate failure reasons or processing issues back to the calling module.
 * Adheres to a standardized format for robust error handling.
 */
data class FailureReason(
    val code: String,       // e.g., "MODEL_LOAD_FAILED", "IMAGE_PROCESSING_FAILED"
    val message: String,    // Human-readable description of the failure.
    val isCritical: Boolean // Indicates if the app should halt or proceed gracefully.
)
