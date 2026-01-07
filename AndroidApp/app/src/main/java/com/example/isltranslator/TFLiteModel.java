package com.example.isltranslator;

import android.content.Context;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.MappedByteBuffer;

public class TFLiteModel {

    private Interpreter interpreter;

    public TFLiteModel(Context context, String modelPath) throws IOException {
        MappedByteBuffer buffer = FileUtil.loadMappedFile(context, modelPath);
        interpreter = new Interpreter(buffer);
    }

    public float[][] predict(float[] input, int numClasses) {
        float[][] output = new float[1][numClasses];
        interpreter.run(input, output);
        return output;
    }

    public void close() {
        if (interpreter != null) interpreter.close();
    }
}
