package com.example.isltranslator;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

public class OverlayView extends View {

    private HandLandmarkerResult result;
    private final Paint pointPaint = new Paint();
    private final Paint linePaint = new Paint();

    // Define connections between landmarks for hand skeleton
    private final int[][] handConnections = {
            {0,1},{1,2},{2,3},{3,4},        // Thumb
            {0,5},{5,6},{6,7},{7,8},        // Index
            {0,9},{9,10},{10,11},{11,12},   // Middle
            {0,13},{13,14},{14,15},{15,16}, // Ring
            {0,17},{17,18},{18,19},{19,20}  // Pinky
    };

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        // Paint for landmarks (points)
        pointPaint.setColor(Color.GREEN);
        pointPaint.setStyle(Paint.Style.FILL);
        pointPaint.setStrokeWidth(8f);

        // Paint for connecting lines (skeleton)
        linePaint.setColor(Color.GREEN);
        linePaint.setStrokeWidth(4f);
    }

    /**
     * Set new hand landmarks result and redraw the overlay
     */
    public void setResults(HandLandmarkerResult result) {
        this.result = result;
        invalidate(); // triggers onDraw
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (result == null || result.landmarks().isEmpty()) return;

        // Loop through each hand detected
        for (int handIndex = 0; handIndex < result.landmarks().size(); handIndex++) {

            // Draw lines (skeleton)
            for (int[] conn : handConnections) {
                int start = conn[0];
                int end = conn[1];

                float startX = result.landmarks().get(handIndex).get(start).x() * getWidth();
                float startY = result.landmarks().get(handIndex).get(start).y() * getHeight();
                float endX = result.landmarks().get(handIndex).get(end).x() * getWidth();
                float endY = result.landmarks().get(handIndex).get(end).y() * getHeight();

                canvas.drawLine(startX, startY, endX, endY, linePaint);
            }

            // Draw points (landmarks)
            for (int i = 0; i < result.landmarks().get(handIndex).size(); i++) {
                float x = result.landmarks().get(handIndex).get(i).x() * getWidth();
                float y = result.landmarks().get(handIndex).get(i).y() * getHeight();
                canvas.drawCircle(x, y, 8f, pointPaint);
            }
        }
    }
}
