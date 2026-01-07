package com.example.isltranslator;

import android.content.Context;

import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

public class HandLandmarkerHelper {

    private final Context context;
    private final HandLandmarkerListener listener;
    private HandLandmarker handLandmarker;

    public HandLandmarkerHelper(Context context, HandLandmarkerListener listener) {
        this.context = context;
        this.listener = listener;
    }

    public void setup() {
        HandLandmarker.HandLandmarkerOptions options =
                HandLandmarker.HandLandmarkerOptions.builder()
                        .setBaseOptions(
                                BaseOptions.builder()
                                        .setModelAssetPath("hand_landmarker.task")
                                        .build()
                        )
                        .setNumHands(2)
                        .setRunningMode(RunningMode.LIVE_STREAM)
                        .setResultListener(
                                (result, inputImage) -> listener.onResults(result)
                        )
                        .build();

        handLandmarker = HandLandmarker.createFromOptions(context, options);
    }

    public void detect(MPImage image) {
        if (handLandmarker != null) {
            handLandmarker.detectAsync(image, System.currentTimeMillis());
        }
    }

    public interface HandLandmarkerListener {
        void onResults(HandLandmarkerResult result);
    }
}
