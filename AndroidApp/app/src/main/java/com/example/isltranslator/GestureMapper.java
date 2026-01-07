import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmark;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

import java.util.List;

public class GestureMapper {

    public static String mapResultToGesture(HandLandmarkerResult result) {
        if (result == null || result.multiHandLandmarks().isEmpty()) return "";

        List<HandLandmark> landmarks = result.multiHandLandmarks().get(0);

        // Example logic for a few letters:
        if (isFist(landmarks)) return "A";
        if (isOpenPalm(landmarks)) return "B";

        return ""; // Default if not recognized
    }

    private static boolean isFist(List<HandLandmark> landmarks) {
        // Thumb tip vs MCP joint, etc.
        // Implement landmark rules for closed fist
        return false;
    }

    private static boolean isOpenPalm(List<HandLandmark> landmarks) {
        // Check if all fingers are extended
        return false;
    }
}
