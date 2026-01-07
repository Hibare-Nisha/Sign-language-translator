package com.example.isltranslator;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import android.util.Log;

public class FrameAnalyzer implements ImageAnalysis.Analyzer {

    @Override
    public void analyze(@NonNull ImageProxy image) {
        // This is where each camera frame comes
        Log.d("FrameAnalyzer", "Frame received: " + image.getWidth() + " x " + image.getHeight());

        // VERY IMPORTANT: close image after processing
        image.close();
    }
}
