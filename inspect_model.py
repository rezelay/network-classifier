import joblib
import numpy as np

mlp    = joblib.load('model/mlp.joblib')
scaler = joblib.load('model/scaler.joblib')

# ── MLP overview ─────────────────────────────────────────────────────────────
print('=== MLP Classifier ===')
print(f'Architecture      : {mlp.n_features_in_} → {mlp.hidden_layer_sizes} → 1')
print(f'Total layers      : {mlp.n_layers_}')
print(f'Activation        : {mlp.activation}')
print(f'Solver            : {mlp.solver}')
print(f'Epochs trained    : {mlp.n_iter_}')
print(f'Final loss        : {mlp.loss_:.6f}')
print(f'Classes           : {mlp.classes_}')

# ── Layer-by-layer weights ────────────────────────────────────────────────────
print('\n=== Layer Weights & Biases ===')
for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
    print(f'  Layer {i+1}: weights {W.shape}  biases {b.shape}'
          f'  |  weight range [{W.min():.4f}, {W.max():.4f}]')

# ── Total parameter count ─────────────────────────────────────────────────────
total_params = sum(W.size + b.size for W, b in zip(mlp.coefs_, mlp.intercepts_))
print(f'\nTotal parameters  : {total_params:,}')

# ── Scaler ────────────────────────────────────────────────────────────────────
print('\n=== Standard Scaler ===')
print(f'Features          : {scaler.n_features_in_}')
print(f'Mean  (first 5)   : {np.round(scaler.mean_[:5], 4)}')
print(f'Std   (first 5)   : {np.round(scaler.scale_[:5], 4)}')
print(f'Mean  range       : [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]')
print(f'Std   range       : [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]')