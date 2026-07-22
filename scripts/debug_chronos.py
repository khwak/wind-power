import torch
import numpy as np

from prepare_data import get_tabular_data
from config import TARGET_COLS, CHRONOS_MODEL_PATH
from chronos import BaseChronosPipeline

print("=" * 80)
print("Loading data...")

train_df, X_train, test_df, X_test, sample_sub = get_tabular_data()

print("train_df shape :", train_df.shape)
print("test_df shape  :", test_df.shape)
print("sample_sub len :", len(sample_sub))
print()

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device :", device)
print("Model  :", CHRONOS_MODEL_PATH)
print()

pipeline = BaseChronosPipeline.from_pretrained(
    CHRONOS_MODEL_PATH,
    device_map=device,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

prediction_length = len(sample_sub)

print("Prediction Length :", prediction_length)
print("=" * 80)

target = TARGET_COLS[0]
print(f"\n===== Target : {target} =====")

past_values = train_df[target].dropna().values[-4000:]

print("past_values length :", len(past_values))
print("NaN count :", np.isnan(past_values).sum())
print("dtype :", past_values.dtype)

context = torch.tensor(
    past_values,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0)

print()
print("context shape :", context.shape)
print("context dtype :", context.dtype)

print("\nRunning prediction...\n")

quantiles, mean = pipeline.predict_quantiles(
    inputs=context,
    prediction_length=prediction_length,
    quantile_levels=[0.5],
)

print("=" * 80)
print("RESULT")
print("=" * 80)

###############################################################
# 타입 확인
###############################################################

print("quantiles type :", type(quantiles))
print("mean type      :", type(mean))

###############################################################
# list인지 확인
###############################################################

if isinstance(quantiles, list):
    print("\nquantiles is LIST")
    print("len(quantiles) =", len(quantiles))

    for i, q in enumerate(quantiles):
        print(f"quantiles[{i}]")
        print(" type :", type(q))
        print(" shape:", q.shape)
        print(" dtype:", q.dtype)

if isinstance(mean, list):
    print("\nmean is LIST")
    print("len(mean) =", len(mean))

    for i, m in enumerate(mean):
        print(f"mean[{i}]")
        print(" type :", type(m))
        print(" shape:", m.shape)
        print(" dtype:", m.dtype)

###############################################################
# 실제 indexing 결과 확인
###############################################################

print("\n" + "=" * 80)
print("INDEX TEST")
print("=" * 80)

if isinstance(quantiles, list):

    q = quantiles[0]

    print("q.shape =", q.shape)

    tests = {
        "q[0]": q[0],
        "q[0,:,0]": q[0,:,0],
        "q[:,0]": q[:,0],
        "q[:,:,0]": q[:,:,0],
    }

    for name, value in tests.items():
        print(f"\n{name}")
        print("shape :", value.shape)
        print("length:", len(value.flatten()))

if isinstance(mean, list):

    m = mean[0]

    print("\nm.shape =", m.shape)

    tests = {
        "m[0]": m[0],
        "m[:,:]": m[:,:],
    }

    for name, value in tests.items():
        print(f"\n{name}")
        print("shape :", value.shape)
        print("length:", len(value.flatten()))

###############################################################
# numpy 변환 테스트
###############################################################

print("\n" + "=" * 80)
print("NUMPY TEST")
print("=" * 80)

if isinstance(mean, list):

    arr = mean[0][0].cpu().numpy()

    print("mean[0][0]")
    print("shape :", arr.shape)
    print("length:", len(arr))
    print("first 5 :", arr[:5])
    print("last 5  :", arr[-5:])

if isinstance(quantiles, list):

    arr = quantiles[0][0,:,0].cpu().numpy()

    print("\nquantiles[0][0,:,0]")
    print("shape :", arr.shape)
    print("length:", len(arr))
    print("first 5 :", arr[:5])
    print("last 5  :", arr[-5:])

###############################################################
# 최종 검증
###############################################################

print("\n" + "=" * 80)
print("FINAL CHECK")
print("=" * 80)

print("prediction_length :", prediction_length)
print("sample_sub length :", len(sample_sub))

if isinstance(mean, list):
    print("mean length       :", len(mean[0][0]))

if isinstance(quantiles, list):
    print("quantile length   :", len(quantiles[0][0,:,0]))

print("=" * 80)