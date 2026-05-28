package com.fishtvai.ml.util

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.media.Image
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class PreprocessedResult(
    val tensorBuffer: ByteBuffer,
    val bitmap: Bitmap
)

object ImageUtils {

    @SuppressLint("NewApi")
    fun processImageToTensor(image: Image, targetWidth: Int, targetHeight: Int): PreprocessedResult? {
        if (image == null) return null

        val yBuffer: ByteBuffer = image.planes[0].buffer
        val uBuffer: ByteBuffer = image.planes[1].buffer
        val vBuffer: ByteBuffer = image.planes[2].buffer
        val width = image.width
        val height = image.height

        val fullSize = width * height
        val rgbBytes = ByteArray(fullSize * 3)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val yIndex = y * width + x
                val uIndex = (y / 2) * (width / 2) + (x / 2)
                val vIndex = uIndex

                val Y = yBuffer.get(yIndex).toInt() and 0xFF
                val U = uBuffer.get(uIndex).toInt() and 0xFF
                val V = vBuffer.get(vIndex).toInt() and 0xFF

                val r = (Y + 1.402 * (V - 128)).toInt().coerceIn(0, 255)
                val g = (Y - 0.344 * (U - 128) - 0.714 * (V - 128)).toInt().coerceIn(0, 255)
                val b = (Y + 1.772 * (U - 128)).toInt().coerceIn(0, 255)

                val idx = yIndex * 3
                rgbBytes[idx] = r.toByte()
                rgbBytes[idx + 1] = g.toByte()
                rgbBytes[idx + 2] = b.toByte()
            }
        }

        val tensorSize = targetWidth * targetHeight * 3
        val tensorBuffer = ByteBuffer.allocateDirect(tensorSize * 4).order(ByteOrder.nativeOrder())

        val numPixels = targetWidth * targetHeight
        val chwData = FloatArray(numPixels * 3)
        val pixels = IntArray(numPixels)

        for (py in 0 until targetHeight) {
            for (px in 0 until targetWidth) {
                val sourceX = (px * width / targetWidth).coerceIn(0, width - 1)
                val sourceY = (py * height / targetHeight).coerceIn(0, height - 1)

                val srcIdx = (sourceY * width + sourceX) * 3
                val r = rgbBytes[srcIdx].toInt() and 0xFF
                val g = rgbBytes[srcIdx + 1].toInt() and 0xFF
                val b = rgbBytes[srcIdx + 2].toInt() and 0xFF

                val pixelIdx = (py * targetWidth + px)
                val hwcIdx = pixelIdx * 3
                chwData[hwcIdx] = r / 255.0f
                chwData[hwcIdx + 1] = g / 255.0f
                chwData[hwcIdx + 2] = b / 255.0f

                pixels[pixelIdx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        val bitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)

        // Write in CHW order (model expects NCHW: [1, 3, H, W])
        for (c in 0 until 3) {
            for (py in 0 until targetHeight) {
                for (px in 0 until targetWidth) {
                    val hwcIdx = (py * targetWidth + px) * 3 + c
                    tensorBuffer.putFloat(chwData[hwcIdx])
                }
            }
        }

        tensorBuffer.rewind()
        return PreprocessedResult(tensorBuffer, bitmap)
    }
}
