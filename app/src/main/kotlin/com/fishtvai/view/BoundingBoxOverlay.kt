package com.fishtvai.view

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.fishtvai.model.Detection

class BoundingBoxOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<Detection> = emptyList()
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private val labelPaint = Paint().apply {
        color = Color.GREEN
        textSize = 36f
        isFakeBoldText = true
    }

    private val bgPaint = Paint().apply {
        color = Color.argb(140, 0, 200, 0)
        style = Paint.Style.FILL
    }

    private val tempRectF = RectF()

    fun setDetections(detections: List<Detection>, imageWidth: Int, imageHeight: Int) {
        this.detections = detections
        this.imageWidth = maxOf(imageWidth, 1)
        this.imageHeight = maxOf(imageHeight, 1)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val scale = minOf(width.toFloat() / imageWidth, height.toFloat() / imageHeight)
        val offsetX = (width - imageWidth * scale) / 2f
        val offsetY = (height - imageHeight * scale) / 2f

        for (detection in detections) {
            val src = detection.boundingBoxPixels

            tempRectF.set(
                src.left * scale + offsetX,
                src.top * scale + offsetY,
                src.right * scale + offsetX,
                src.bottom * scale + offsetY
            )

            canvas.drawRoundRect(tempRectF, 8f, 8f, boxPaint)

            val label = "${detection.label} ${"%.0f".format(detection.confidence * 100)}%"
            val textWidth = labelPaint.measureText(label)
            val labelRect = Rect(
                tempRectF.left.toInt(),
                (tempRectF.top - 40).toInt(),
                (tempRectF.left + textWidth + 12).toInt(),
                tempRectF.top.toInt()
            )

            canvas.drawRect(labelRect, bgPaint)
            canvas.drawText(label, tempRectF.left + 6, tempRectF.top - 10, labelPaint)
        }
    }
}
