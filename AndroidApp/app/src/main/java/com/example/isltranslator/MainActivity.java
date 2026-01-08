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
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity implements HandLandmarkerHelper.HandLandmarkerListener {

    private PreviewView previewView;
    private OverlayView overlayView;
    private TextView tvResult;

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

        previewView = findViewById(R.id.previewView);
        overlayView = findViewById(R.id.overlayView);
        tvResult = findViewById(R.id.tvResult);

        try {
            tfliteModel = new TFLiteModel(this, "isl_model.tflite");
        } catch (IOException e) {
            e.printStackTrace();
        }

        handLandmarkerHelper = new HandLandmarkerHelper(this, this);
        handLandmarkerHelper.setup();

        cameraExecutor = Executors.newSingleThreadExecutor();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_CODE);
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                CameraSelector cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA;

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor,
                        new FrameAnalyzer(handLandmarkerHelper));

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

    private float[] createFeatureVector(HandLandmarkerResult result) {
        if (result == null || result.landmarks().isEmpty()) return new float[63];

        List<NormalizedLandmark> landmarks = result.landmarks().get(0);
        float[] vector = new float[63];

        for (int i = 0; i < 21; i++) {
            vector[i * 3]     = landmarks.get(i).x();
            vector[i * 3 + 1] = landmarks.get(i).y();
            vector[i * 3 + 2] = landmarks.get(i).z();
        }

        return vector;
    }

    @Override
    public void onResults(HandLandmarkerResult result) {

        if (result == null || result.landmarks().isEmpty()) {
            runOnUiThread(() -> {
                overlayView.setResults(null);
                tvResult.setText("");
            });
            return;
        }

        float[] inputVector = createFeatureVector(result);
        float[][] output = tfliteModel.predict(inputVector, 26);

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

        runOnUiThread(() -> {
            tvResult.setText(predictedLetter);
            overlayView.setResults(result);
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (tfliteModel != null) tfliteModel.close();
        if (cameraExecutor != null) cameraExecutor.shutdown();
    }
}
