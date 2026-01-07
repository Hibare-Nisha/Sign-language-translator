package com.example.isltranslator;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmark;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity implements HandLandmarkerHelper.HandLandmarkerListener {

    private PreviewView previewView;
    private OverlayView overlayView;
    private TextView gestureTextView;

    private TFLiteModel tfliteModel;
    private HandLandmarkerHelper handLandmarkerHelper;

    private ExecutorService cameraExecutor;

    private static final int CAMERA_PERMISSION_CODE = 100;

    private final String[] alphabetLabels = {
            "A","B","C","D","E","F","G","H","I","J","K","L","M",
            "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize UI
        previewView = findViewById(R.id.previewView);
        overlayView = findViewById(R.id.overlayView);
        gestureTextView = findViewById(R.id.gestureTextView);

        // Load TFLite model
        try {
            tfliteModel = new TFLiteModel(this, "isl_model.tflite");
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Setup HandLandmarker
        handLandmarkerHelper = new HandLandmarkerHelper(this, this);
        handLandmarkerHelper.setup();

        // Executor for CameraX image analysis
        cameraExecutor = Executors.newSingleThreadExecutor();

        // Request camera permission if not granted
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_CODE);
        }
    }

    /**
     * Start CameraX with Preview + ImageAnalysis
     */
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                // Preview
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // Front camera
                CameraSelector cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA;

                // Image analysis for frames
                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor,
                        new FrameAnalyzer(handLandmarkerHelper));

                // Bind to lifecycle
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(
                        this,
                        cameraSelector,
                        preview,
                        imageAnalysis
                );

            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            }
        }
    }

    /**
     * Convert HandLandmarkerResult to TFLite input vector
     */
    private float[] createFeatureVector(HandLandmarkerResult result) {
        if (result == null || result.multiHandLandmarks().isEmpty()) return new float[63];

        List<HandLandmark> landmarks = result.multiHandLandmarks().get(0);
        float[] vector = new float[21 * 3]; // x, y, z

        for (int i = 0; i < 21; i++) {
            vector[i * 3 + 0] = landmarks.get(i).x();
            vector[i * 3 + 1] = landmarks.get(i).y();
            vector[i * 3 + 2] = landmarks.get(i).z();
        }

        return vector;
    }

    /**
     * Called when HandLandmarker returns results
     */
    @Override
    public void onResults(HandLandmarkerResult result) {
        if (result != null && !result.multiHandLandmarks().isEmpty()) {

            // Convert landmarks to vector
            float[] inputVector = createFeatureVector(result);

            // Run TFLite model
            float[][] output = tfliteModel.predict(inputVector, 26);

            // Find max probability
            int maxIdx = 0;
            float maxVal = output[0][0];
            for (int i = 1; i < output[0].length; i++) {
                if (output[0][i] > maxVal) {
                    maxVal = output[0][i];
                    maxIdx = i;
                }
            }

            String predictedLetter = alphabetLabels[maxIdx];
            Log.d("HAND", "Predicted: " + predictedLetter);

            // Update UI
            runOnUiThread(() -> {
                gestureTextView.setText(predictedLetter);
                overlayView.setResults(result);
            });

        } else {
            // Clear overlay and text if no hand
            runOnUiThread(() -> {
                overlayView.setResults(null);
                gestureTextView.setText("");
            });
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        // Release TFLite model resources
        if (tfliteModel != null) tfliteModel.close();

        // Shutdown camera executor
        if (cameraExecutor != null) cameraExecutor.shutdown();
    }
}
