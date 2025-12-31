#Imports
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# CONFIGURE
DATASET_DIR = "D:\My Skills\majorproj\databook"   # <-- folder that contains A, B, C, ....
SEQ_LEN = 90                # fixed length for GRU
FEATURE_DIM = 159           # from your .npy files


#PAD / TRUNCATE FUNCTION (CRITICAL)
def pad_or_truncate(sequence, target_len=SEQ_LEN):
    current_len = sequence.shape[0]

    if current_len > target_len:
        return sequence[:target_len]

    if current_len < target_len:
        pad = np.zeros((target_len - current_len, sequence.shape[1]))
        return np.vstack((sequence, pad))

    return sequence

# load dataset

X = []
y = []
label_map = {}

current_label = 0

for class_name in sorted(os.listdir(DATASET_DIR)):
    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    label_map[current_label] = class_name
    print(f"Loading class '{class_name}' → label {current_label}")

    for file in os.listdir(class_path):
        if file.endswith(".npy"):
            file_path = os.path.join(class_path, file)
            sequence = np.load(file_path)

            # Safety check
            if sequence.shape != (SEQ_LEN, FEATURE_DIM ):
                print(f"⚠ Skipping {file} due to wrong shape {sequence.shape}")
                continue

            X.append(sequence)
            y.append(current_label)

    current_label += 1

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

NUM_CLASSES = len(label_map)

print("\nDataset loaded successfully")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", label_map)


# TRAIN–TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
X = []
y = []
label_map = {}
current_label = 0

print("Dataset dir:", DATASET_DIR)

for class_name in sorted(os.listdir(DATASET_DIR)):
    class_path = os.path.join(DATASET_DIR, class_name)
    print("\nChecking class folder:", class_path)

    if not os.path.isdir(class_path):
        print("  ❌ Not a directory")
        continue

    files = os.listdir(class_path)
    print("  Files found:", files)

    label_map[current_label] = class_name

    for file in files:
        print("    Reading:", file)

        file_path = os.path.join(class_path, file)

        try:
            data = np.load(file_path)
            print("      Shape:", data.shape)

            X.append(data)
            y.append(current_label)

        except Exception as e:
            print("      ❌ Failed to load:", e)

    current_label += 1

print("\nFINAL RESULT")
print("Total samples:", len(X))



# CREATE 3-LAYER GRU MODEL

def create_gru_model():
    model = models.Sequential([
        layers.Input(shape=(SEQ_LEN, FEATURE_DIM )),

        layers.GRU(64, return_sequences=True),
        layers.GRU(32, return_sequences=True),
        layers.GRU(16),

        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# TRAIN MODEL

model = create_gru_model()
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    shuffle=True
)

# Test model

loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {acc:.4f}")

# SAVE MODEL

model.save("signlanguage_model.h5")
np.save("label_map.npy", label_map)

print("\n✅ Model saved as gru_sign_language_model.h5")
print("✅ Labels saved as label_map.npy")


