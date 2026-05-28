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

        val planes = image.planes
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride

        val width = image.width
        val height = image.height

        val isSemiPlanar = uPixelStride == 2

        // Copy planes to byte arrays to avoid buffer position issues
        val yArr = ByteArray(yBuf.remaining())
        yBuf.get(yArr)
        val uArr = ByteArray(uBuf.remaining())
        uBuf.get(uArr)
        val vArr = ByteArray(vBuf.remaining())
        vBuf.get(vArr)

        val rgbBytes = ByteArray(width * height * 3)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val yIdx = y * yRowStride + x * yPixelStride
                val Y = yArr[yIdx].toInt() and 0xFF

                val uvY = y / 2
                val uvX = x / 2
                val uvIdx = uvY * uRowStride + uvX * uPixelStride

                val U: Int
                val V: Int
                if (isSemiPlanar) {
                    U = uArr[uvIdx].toInt() and 0xFF
                    V = if (uvIdx + 1 < uArr.size) uArr[uvIdx + 1].toInt() and 0xFF else 128
                } else {
                    val vIdx = uvY * vRowStride + uvX * vPixelStride
                    U = uArr[uvIdx].toInt() and 0xFF
                    V = if (vIdx < vArr.size) vArr[vIdx].toInt() and 0xFF else 128
                }

                val r = (Y + 1.402 * (V - 128)).toInt().coerceIn(0, 255)
                val g = (Y - 0.344 * (U - 128) - 0.714 * (V - 128)).toInt().coerceIn(0, 255)
                val b = (Y + 1.772 * (U - 128)).toInt().coerceIn(0, 255)

                val idx = (y * width + x) * 3
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
