"""
MIT License

Copyright (c) 2026 Jernej Hribar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import time
import math
import csv
import warnings
import numpy as np
import h5py
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

#  Global constants & config

BASE_GRID_H = 9
BASE_GRID_W = 11
SCALE_FACTOR = 1  # overwritten in main loop

GLOBAL_MEAN = None
GLOBAL_STD = None

SAVE_DIR = "saved_models"
PLOTS_DIR = "plots"
RESULTS_DIR = "results_csv"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FORCE_RETRAIN = False

tf.random.set_seed(0)
np.random.seed(0)

#  Environment based on the dataset: An Indoor Radio Mapping Dataset Combining 3D Point Clouds and RSSI


class SITEnvironment:
    """
    Environment for RSSI data captured at SIT.
    Builds a completed base 9x11 RSSI grid, then upsamples by SCALE_FACTOR.
    """
    def __init__(self, scenario_id=0, h5_path="data/WiFi_RSSI_data2025-06-25_11-32-25.h5"):
        self.h5_path = h5_path
        self.dataset = h5py.File(self.h5_path, "r")
        self.scenario_id = None
        self.changeScenario(scenario_id)

    def changeScenario(self, scenario_id: int):
        global SCALE_FACTOR

        self.scenario_id = scenario_id
        self.data_group = self.dataset["data"][self.scenario_id]
        self.setup = self.dataset["setup"][self.scenario_id]
        self.matrix_ind = self.dataset["indices"][0]
        self.ap_location = self.dataset["ap_locations"][self.scenario_id]

        room_data = self.data_group[0:BASE_GRID_H, 0:BASE_GRID_W]
        H0, W0 = room_data.shape

        x = np.linspace(0, W0 - 1, W0)
        y = np.linspace(0, H0 - 1, H0)
        X, Y = np.meshgrid(x, y)

        mask_missing = (room_data == 0.0)
        points_valid = np.column_stack((X[~mask_missing], Y[~mask_missing]))
        values_valid = room_data[~mask_missing]

        Z_filled = griddata(points_valid, values_valid, (X, Y), method="linear")
        mask_unfilled = np.isnan(Z_filled)
        if np.any(mask_unfilled):
            Z_filled[mask_unfilled] = griddata(points_valid, values_valid, (X, Y), method="nearest")[mask_unfilled]

        Z_filled = Z_filled.astype(np.float32)

        if SCALE_FACTOR <= 1:
            self.Z_filled = Z_filled
        else:
            H_new = H0 * SCALE_FACTOR
            W_new = W0 * SCALE_FACTOR
            z_tf = tf.convert_to_tensor(Z_filled[None, ..., None], dtype=tf.float32)
            z_up = tf.image.resize(z_tf, [H_new, W_new], method="bilinear")
            self.Z_filled = z_up.numpy()[0, :, :, 0].astype(np.float32)


#  Baseline we use to compare World Models approach to

def fit_gp_kernel_on_static(env: SITEnvironment, static_sid: int, target_points: int = 500, random_state: int = 0):
    env.changeScenario(static_sid)
    Z = np.array(env.Z_filled, dtype=np.float64)
    H, W = Z.shape

    total = H * W
    stride = int(np.ceil(np.sqrt(total / float(target_points))))
    stride = max(1, stride)

    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    XX, YY = np.meshgrid(xs, ys)
    X = np.column_stack((XX.ravel(), YY.ravel()))
    y = Z[YY, XX].ravel()

    kernel = (
        C(1.0, (1e-3, 1e3)) *
        RBF(length_scale=2.5, length_scale_bounds=(1e-2, 1e2)) +
        WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 1e1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=random_state
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        print(f"[GP] Fitting kernel on static {static_sid} with {X.shape[0]} subsampled points (scale={SCALE_FACTOR})")
        gp.fit(X, y)

    print(f"[GP] Learned kernel (static {static_sid}, scale={SCALE_FACTOR}): {gp.kernel_}")
    return gp.kernel_


def gp_predict_occupied(env: SITEnvironment, occ_sid: int, coords, kernel_static):
    env.changeScenario(occ_sid)
    Z_occ = np.array(env.Z_filled, dtype=np.float64)
    H, W = Z_occ.shape

    XX, YY = np.meshgrid(np.arange(W), np.arange(H))
    X_test = np.column_stack((XX.ravel(), YY.ravel())).astype(np.float64)

    if len(coords) == 0:
        mu = np.zeros_like(Z_occ, dtype=np.float64)
        rmse = float(np.sqrt(np.mean((mu - Z_occ) ** 2)))
        mae = float(np.mean(np.abs(mu - Z_occ)))
        return rmse, mae, mu.astype(np.float32)

    X_train = np.array([[x, y] for (x, y) in coords], dtype=np.float64)
    y_train = np.array([Z_occ[y, x] for (x, y) in coords], dtype=np.float64)

    gp = GaussianProcessRegressor(
        kernel=kernel_static,
        alpha=1e-6,
        normalize_y=True,
        optimizer=None,
        random_state=0
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gp.fit(X_train, y_train)

    mu, _ = gp.predict(X_test, return_std=True)
    mu = mu.reshape(H, W)

    rmse = float(np.sqrt(np.mean((mu - Z_occ) ** 2)))
    mae = float(np.mean(np.abs(mu - Z_occ)))
    return rmse, mae, mu.astype(np.float32)


#  Measurement patterns

def maxdist_select(mask: np.ndarray):
    H, W = mask.shape
    meas_coords = [(x, y) for y in range(H) for x in range(W) if mask[y, x]]
    unmeas_coords = [(x, y) for y in range(H) for x in range(W) if not mask[y, x]]

    if len(unmeas_coords) == 0:
        return None
    if len(meas_coords) == 0:
        return (W // 2, H // 2)

    best_coord, best_dist = None, -1.0
    for (ux, uy) in unmeas_coords:
        mind = min(math.hypot(ux - mx, uy - my) for (mx, my) in meas_coords)
        if mind > best_dist:
            best_dist = mind
            best_coord = (ux, uy)
    return best_coord


def generate_measurement_pattern(env: SITEnvironment, scenario_id: int, N_max: int,
                                 strategy: str = "farthest", seed: int = 0):
    env.changeScenario(scenario_id)
    H, W = env.Z_filled.shape
    rng = np.random.default_rng(seed)

    if N_max <= 0:
        return []

    if strategy == "random":
        all_cells = [(x, y) for y in range(H) for x in range(W)]
        rng.shuffle(all_cells)
        return all_cells[:N_max]

    if strategy == "farthest":
        coords = []
        mask = np.zeros((H, W), dtype=bool)
        cx, cy = W // 2, H // 2
        coords.append((cx, cy))
        mask[cy, cx] = True
        while len(coords) < N_max:
            sel = maxdist_select(mask)
            if sel is None:
                break
            x, y = sel
            coords.append((x, y))
            mask[y, x] = True
        return coords

    raise ValueError(f"Unknown strategy {strategy}")


#  Sparse Encoding: Input tensor builder 


def build_input_tensor(env: SITEnvironment, empty_sid: int, occ_sid: int, coords):
    """
    Observation x_t:
      ch0: empty map (normalized)
      ch1: measured values from occupied (normalized) at coords
      ch2: mask 1 at coords
    Target:
      occupied map (normalized) as (H,W,1)
    """
    global GLOBAL_MEAN, GLOBAL_STD

    env.changeScenario(empty_sid)
    Z_empty_raw = np.array(env.Z_filled, dtype=np.float32)

    env.changeScenario(occ_sid)
    Z_occ_raw = np.array(env.Z_filled, dtype=np.float32)
    H, W = Z_occ_raw.shape

    if GLOBAL_MEAN is not None and GLOBAL_STD is not None:
        Z_empty_norm = (Z_empty_raw - GLOBAL_MEAN) / GLOBAL_STD
        Z_occ_norm = (Z_occ_raw - GLOBAL_MEAN) / GLOBAL_STD
    else:
        Z_empty_norm = Z_empty_raw
        Z_occ_norm = Z_occ_raw

    meas_map = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)
    for (x, y) in coords:
        meas_map[y, x] = Z_occ_norm[y, x]
        mask[y, x] = 1.0

    obs = np.stack([Z_empty_norm, meas_map, mask], axis=-1)  # (H,W,3)
    target = Z_occ_norm[..., None]                           # (H,W,1)
    return obs, target, Z_empty_raw, Z_occ_raw


#  WORLD MODELS: VAE (Variational Decoder) and Dynamics model


class VisionVAE(tf.keras.Model):
    """
    V: conditional VAE (shape-safe decoder).
    """
    def __init__(self, H: int, W: int, z_dim: int = 64, base_channels: int = 32):
        super().__init__()
        self.H = int(H)
        self.W = int(W)
        self.z_dim = int(z_dim)
        self.base_channels = int(base_channels)

        # Encoder
        self.enc = models.Sequential([
            layers.Input(shape=(self.H, self.W, 3)),
            layers.Conv2D(base_channels, 3, activation="relu", padding="same"),
            layers.Conv2D(base_channels, 3, activation="relu", padding="same"),
            layers.MaxPool2D(pool_size=2),
            layers.Conv2D(base_channels * 2, 3, activation="relu", padding="same"),
            layers.Conv2D(base_channels * 2, 3, activation="relu", padding="same"),
            layers.MaxPool2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
        ])
        self.z_mean = layers.Dense(self.z_dim)
        self.z_logvar = layers.Dense(self.z_dim)

        # Decoder: ceil base grid and resize to exact output
        self.h0 = max(1, int(math.ceil(self.H / 4.0)))
        self.w0 = max(1, int(math.ceil(self.W / 4.0)))
        self.dec_ch = base_channels * 2

        self.dec_dense = layers.Dense(self.h0 * self.w0 * self.dec_ch, activation="relu")
        self.dec_reshape = layers.Reshape((self.h0, self.w0, self.dec_ch))

        self.dec = models.Sequential([
            layers.Conv2D(self.dec_ch, 3, activation="relu", padding="same"),
            layers.UpSampling2D(size=2, interpolation="nearest"),
            layers.Conv2D(base_channels, 3, activation="relu", padding="same"),
            layers.UpSampling2D(size=2, interpolation="nearest"),
            layers.Conv2D(base_channels, 3, activation="relu", padding="same"),
            layers.Conv2D(1, 3, activation="linear", padding="same"),
        ])

    @staticmethod
    def reparameterize(z_mean, z_logvar):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps

    def encode(self, x, training=False):
        h = self.enc(x, training=training)
        return self.z_mean(h), self.z_logvar(h)

    def decode(self, z, training=False):
        h = self.dec_dense(z, training=training)
        h = self.dec_reshape(h)
        y = self.dec(h, training=training)
        if (y.shape[1] != self.H) or (y.shape[2] != self.W):
            y = tf.image.resize(y, [self.H, self.W], method="bilinear")
        return y

    def call(self, x, training=False):
        z_mean, z_logvar = self.encode(x, training=training)
        z = self.reparameterize(z_mean, z_logvar) if training else z_mean
        y = self.decode(z, training=training)
        return y, z_mean, z_logvar


class LatentDynamics(tf.keras.Model):
    """
    M: action-conditioned latent dynamics model.

    CRITICAL FIX:
      - Provide a `call()` method so Keras can build the model via dyn(...)
      - Keras may not mark the model built if only a custom method (step) is used.
    """
    def __init__(self, z_dim: int, hidden_dim: int = 128, act_dim: int = 2):
        super().__init__()
        self.z_dim = int(z_dim)
        self.hidden_dim = int(hidden_dim)
        self.act_dim = int(act_dim)

        self.act_embed = models.Sequential([
            layers.Input(shape=(self.act_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
        ])

        self.lstm = layers.LSTMCell(self.hidden_dim)
        self.out = models.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
        ])
        self.z_mean_next = layers.Dense(self.z_dim)
        self.z_logvar_next = layers.Dense(self.z_dim)

    @staticmethod
    def reparameterize(z_mean, z_logvar):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps

    def init_state(self, batch_size: int):
        h0 = tf.zeros((batch_size, self.hidden_dim), dtype=tf.float32)
        c0 = tf.zeros((batch_size, self.hidden_dim), dtype=tf.float32)
        return [h0, c0]

    def step(self, z_t, a_t, state, training=False, sample=False):
        a_emb = self.act_embed(a_t, training=training)
        x = tf.concat([z_t, a_emb], axis=-1)
        h, new_state = self.lstm(x, state, training=training)
        u = self.out(h, training=training)
        mu = self.z_mean_next(u)
        lv = self.z_logvar_next(u)
        z_next = self.reparameterize(mu, lv) if (training or sample) else mu
        return z_next, mu, lv, new_state

    def call(self, inputs, training=False, sample=False):
        """
        Build-friendly call:
          inputs = (z_t, a_t, h, c)  OR (z_t, a_t) with default zero state

        Returns:
          z_next, mu, logvar, (h_next, c_next)
        """
        if not isinstance(inputs, (tuple, list)):
            raise ValueError("LatentDynamics.call expects inputs tuple/list.")

        if len(inputs) == 2:
            z_t, a_t = inputs
            state = self.init_state(tf.shape(z_t)[0])
        elif len(inputs) == 4:
            z_t, a_t, h, c = inputs
            state = [h, c]
        else:
            raise ValueError("LatentDynamics.call expects (z_t,a_t) or (z_t,a_t,h,c).")

        return self.step(z_t, a_t, state, training=training, sample=sample)


#  Saving and Loading of trained Models, 


def ensure_built_vae(vae: VisionVAE):
    dummy = tf.zeros((1, vae.H, vae.W, 3), dtype=tf.float32)
    _ = vae(dummy, training=False)


def ensure_built_dyn(dyn: LatentDynamics):
    """
    Robust across TF/Keras versions:
      - call dyn(...) (which uses call()) to force variable creation
    """
    dummy_z = tf.zeros((1, dyn.z_dim), dtype=tf.float32)
    dummy_a = tf.zeros((1, 2), dtype=tf.float32)
    _ = dyn((dummy_z, dummy_a), training=False, sample=False)

def weights_path_vae(scale_factor: int) -> str:
    return os.path.join(SAVE_DIR, f"WM_VisionVAE_sf{scale_factor:02d}.weights.h5")

def weights_path_dyn(scale_factor: int) -> str:
    return os.path.join(SAVE_DIR, f"WM_Dynamics_sf{scale_factor:02d}.weights.h5")

def load_or_train_two_models(scale_factor: int, vae: VisionVAE, dyn: LatentDynamics,
                             train_fn, train_kwargs: dict):
    """
    Ensures models are built before load/save.
    """
    pV = weights_path_vae(scale_factor)
    pM = weights_path_dyn(scale_factor)

    ensure_built_vae(vae)
    ensure_built_dyn(dyn)

    if (not FORCE_RETRAIN) and os.path.exists(pV) and os.path.exists(pM):
        vae.load_weights(pV)
        dyn.load_weights(pM)
        print(f"[Load] Using existing weights:\n  V: {pV}\n  M: {pM}")
        return False

    print(f"[Train] Training World Models (VAE + Dynamics) for scale={scale_factor}")
    train_fn(vae=vae, dyn=dyn, **train_kwargs)

    # Build again (harmless, ensures built flag is set)
    ensure_built_vae(vae)
    ensure_built_dyn(dyn)

    vae.save_weights(pV)
    dyn.save_weights(pM)
    print(f"[Save] Weights saved:\n  V: {pV}\n  M: {pM}")
    return True


#  Training (VAE + Dynamics)


def kl_gaussian(mu, logvar):
    return 0.5 * tf.reduce_sum(tf.exp(logvar) + tf.square(mu) - 1.0 - logvar, axis=-1)

def nll_diag_gaussian(z_true, mu, logvar):
    return 0.5 * tf.reduce_sum(logvar + tf.square(z_true - mu) * tf.exp(-logvar), axis=-1)

def train_world_models(env: SITEnvironment,
                       vae: VisionVAE,
                       dyn: LatentDynamics,
                       train_pairs,
                       N_choices=(0, 1, 3, 5, 10, 15, 20, 25),
                       epochs=200,
                       episodes_per_epoch=50,
                       pattern_strategy="random",
                       lr=1e-3,
                       beta_kl=1e-3,
                       seed=0):
    """
    Train:
      - VAE reconstruction + KL on x_t -> occupied map
      - Latent one-step dynamics: predict z_{t+1} from (z_t, a_t)
    """
    optV = tf.keras.optimizers.Adam(lr)
    optM = tf.keras.optimizers.Adam(lr)
    rng = np.random.default_rng(seed)

    for epoch in range(int(epochs)):
        lossesV, lossesM = [], []

        for _ in range(int(episodes_per_epoch)):
            empty_sid, occ_sid = train_pairs[rng.integers(len(train_pairs))]
            N = int(rng.choice(N_choices))
            pat_seed = int(rng.integers(1_000_000_000))

            coords_full = generate_measurement_pattern(env, occ_sid, N_max=max(1, N),
                                                       strategy=pattern_strategy, seed=pat_seed)
            coords_full = coords_full[:max(1, N)] if N > 0 else []

            env.changeScenario(occ_sid)
            H, W = env.Z_filled.shape

            if len(coords_full) == 0:
                coords_t = []
                a = (int(rng.integers(W)), int(rng.integers(H)))
                coords_tp1 = [a]
            else:
                t = int(rng.integers(0, len(coords_full)))
                coords_t = coords_full[:t]
                if t < len(coords_full):
                    a = coords_full[t]
                else:
                    chosen = set(coords_full)
                    a = (int(rng.integers(W)), int(rng.integers(H)))
                    while a in chosen:
                        a = (int(rng.integers(W)), int(rng.integers(H)))
                coords_tp1 = coords_t + [a]

            obs_t, target, _, _ = build_input_tensor(env, empty_sid, occ_sid, coords_t)
            obs_tp1, _, _, _ = build_input_tensor(env, empty_sid, occ_sid, coords_tp1)

            obs_t_tf = tf.convert_to_tensor(obs_t[None, ...], dtype=tf.float32)
            obs_tp1_tf = tf.convert_to_tensor(obs_tp1[None, ...], dtype=tf.float32)
            target_tf = tf.convert_to_tensor(target[None, ...], dtype=tf.float32)

            ax, ay = a
            a_norm = tf.convert_to_tensor([[ax / max(1, (W - 1)), ay / max(1, (H - 1))]], dtype=tf.float32)

            # ---- Train V ----
            with tf.GradientTape() as tapeV:
                y_hat, z_mu_t, z_lv_t = vae(obs_t_tf, training=True)
                rec = tf.reduce_mean(tf.square(y_hat - target_tf), axis=[1, 2, 3])
                kl = kl_gaussian(z_mu_t, z_lv_t)
                lossV = tf.reduce_mean(rec + beta_kl * kl)

            gV = tapeV.gradient(lossV, vae.trainable_variables)
            gV = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in gV]
            optV.apply_gradients(zip(gV, vae.trainable_variables))
            lossesV.append(float(lossV.numpy()))

            # ---- Train M ----
            state = dyn.init_state(batch_size=1)
            with tf.GradientTape() as tapeM:
                z_mu_t, _ = vae.encode(obs_t_tf, training=False)
                z_t = z_mu_t
                z_mu_tp1, _ = vae.encode(obs_tp1_tf, training=False)
                z_tp1 = z_mu_tp1

                # IMPORTANT: go through dyn(...) so variables are definitely created/used
                _, mu_pred, lv_pred, _ = dyn((z_t, a_norm, state[0], state[1]), training=True, sample=False)
                lossM = tf.reduce_mean(nll_diag_gaussian(z_tp1, mu_pred, lv_pred))

            gM = tapeM.gradient(lossM, dyn.trainable_variables)
            gM = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in gM]
            optM.apply_gradients(zip(gM, dyn.trainable_variables))
            lossesM.append(float(lossM.numpy()))

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"[WM] Epoch {epoch+1}/{epochs} (scale={SCALE_FACTOR}) "
                  f"V_loss={np.mean(lossesV):.4f} M_nll={np.mean(lossesM):.4f}")


#  Evaluation


def vae_predict_occupied(env: SITEnvironment, vae: VisionVAE,
                         empty_sid: int, occ_sid: int, coords):
    global GLOBAL_MEAN, GLOBAL_STD

    obs, _, _, Z_occ_raw = build_input_tensor(env, empty_sid, occ_sid, coords)
    obs_tf = tf.convert_to_tensor(obs[None, ...], dtype=tf.float32)

    y_hat, _, _ = vae(obs_tf, training=False)
    pred_norm = y_hat.numpy()[0, :, :, 0]

    if GLOBAL_MEAN is not None and GLOBAL_STD is not None:
        pred_raw = pred_norm * GLOBAL_STD + GLOBAL_MEAN
    else:
        pred_raw = pred_norm

    rmse = float(np.sqrt(np.mean((pred_raw - Z_occ_raw) ** 2)))
    mae = float(np.mean(np.abs(pred_raw - Z_occ_raw)))
    return rmse, mae, pred_raw.astype(np.float32)


def finetune_vae_on_coords(env: SITEnvironment, vae: VisionVAE,
                           empty_sid: int, occ_sid: int, coords,
                           steps_ft: int = 80, lr_ft: float = 5e-4, beta_kl: float = 1e-3):
    obs, target, _, _ = build_input_tensor(env, empty_sid, occ_sid, coords)
    obs_tf = tf.convert_to_tensor(obs[None, ...], dtype=tf.float32)
    target_tf = tf.convert_to_tensor(target[None, ...], dtype=tf.float32)

    opt = tf.keras.optimizers.Adam(lr_ft)

    for _ in range(int(steps_ft)):
        with tf.GradientTape() as tape:
            y_hat, z_mu, z_lv = vae(obs_tf, training=True)
            rec = tf.reduce_mean(tf.square(y_hat - target_tf), axis=[1, 2, 3])
            kl = kl_gaussian(z_mu, z_lv)
            loss = tf.reduce_mean(rec + beta_kl * kl)
        g = tape.gradient(loss, vae.trainable_variables)
        g = [tf.clip_by_norm(gg, 5.0) if gg is not None else None for gg in g]
        opt.apply_gradients(zip(g, vae.trainable_variables))


def clone_vae(vae: VisionVAE):
    v2 = VisionVAE(vae.H, vae.W, z_dim=vae.z_dim, base_channels=vae.base_channels)
    ensure_built_vae(v2)
    v2.set_weights(vae.get_weights())
    return v2


#  Dreaming-based acquisition


def dream_rollout_uncertainty(env: SITEnvironment,
                              vae: VisionVAE,
                              dyn: LatentDynamics,
                              empty_sid: int,
                              occ_sid: int,
                              current_coords,
                              candidate_coord,
                              dream_samples: int = 12):
    """
    One-step dream planning score:
      encode x_t -> z_t, apply action a_t -> sample z_{t+1}, decode -> maps, variance score
    """
    env.changeScenario(occ_sid)
    H, W = env.Z_filled.shape

    obs_t, _, _, _ = build_input_tensor(env, empty_sid, occ_sid, current_coords)
    obs_t_tf = tf.convert_to_tensor(obs_t[None, ...], dtype=tf.float32)

    z_mu_t, _ = vae.encode(obs_t_tf, training=False)
    z_t = z_mu_t

    ax, ay = candidate_coord
    a_norm = tf.convert_to_tensor([[ax / max(1, (W - 1)), ay / max(1, (H - 1))]], dtype=tf.float32)

    state = dyn.init_state(batch_size=1)

    preds = []
    for _ in range(int(dream_samples)):
        z_next, _, _, _ = dyn((z_t, a_norm, state[0], state[1]), training=False, sample=True)
        y_hat = vae.decode(z_next, training=False).numpy()[0, :, :, 0]
        preds.append(y_hat)

    preds = np.stack(preds, axis=0)
    var_map = preds.var(axis=0)
    return float(var_map.mean())


def generate_measurement_pattern_dream(env: SITEnvironment,
                                       vae: VisionVAE,
                                       dyn: LatentDynamics,
                                       empty_sid: int,
                                       occ_sid: int,
                                       N_max: int,
                                       seed=0,
                                       candidate_pool=40,
                                       dream_samples=12):
    rng = np.random.default_rng(seed)
    env.changeScenario(occ_sid)
    H, W = env.Z_filled.shape

    coords = []
    all_cells = [(x, y) for y in range(H) for x in range(W)]

    for _ in range(int(N_max)):
        chosen = set(coords)
        avail = [c for c in all_cells if c not in chosen]
        if not avail:
            break

        k = min(int(candidate_pool), len(avail))
        cand_idx = rng.choice(len(avail), size=k, replace=False)
        candidates = [avail[i] for i in cand_idx]

        best_c, best_s = None, 1e18
        for c in candidates:
            s = dream_rollout_uncertainty(
                env=env, vae=vae, dyn=dyn,
                empty_sid=empty_sid, occ_sid=occ_sid,
                current_coords=coords, candidate_coord=c,
                dream_samples=dream_samples
            )
            if s < best_s:
                best_s, best_c = s, c

        coords.append(best_c)

    return coords


#  Evaluation and .CSV export


def summarize_metrics_to_rows(metrics, methods, N_list):
    rows = []
    for m in methods:
        for N in N_list:
            rmse_arr = np.array(metrics[m]["rmse"][N], dtype=np.float32)
            mae_arr = np.array(metrics[m]["mae"][N], dtype=np.float32)
            if rmse_arr.size == 0:
                continue
            rows.append({
                "method": m,
                "N": int(N),
                "rmse_mean": float(rmse_arr.mean()),
                "rmse_std": float(rmse_arr.std()),
                "mae_mean": float(mae_arr.mean()),
                "mae_std": float(mae_arr.std()),
                "count": int(rmse_arr.size),
                "scale_factor": int(SCALE_FACTOR),
            })
    return rows


def write_summary_csv(rows, out_path):
    if not rows:
        print(f"[CSV] No rows to write: {out_path}")
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[CSV] Wrote: {out_path}")


def evaluate_worldmodel_vs_gp(env: SITEnvironment,
                              vae_base: VisionVAE,
                              dyn_base: LatentDynamics,
                              kernels_static: dict,
                              pairs,
                              N_list,
                              pattern_strategy="farthest",
                              n_runs=5,
                              steps_ft=80,
                              lr_ft=5e-4,
                              do_finetune=True,
                              dream_cfg=None):
    if dream_cfg is None:
        dream_cfg = dict(candidate_pool=40, dream_samples=12)

    methods = ["copy-empty", "gp", "worldmodel", "worldmodel-dream"]
    N_list = sorted(list(N_list))
    metrics = {m: {"rmse": {N: [] for N in N_list}, "mae": {N: [] for N in N_list}} for m in methods}

    print(f"\n=== Evaluation (scale={SCALE_FACTOR}) ===")
    print("Pairs:", pairs)
    print("N_list:", N_list)

    for (empty_sid, occ_sid) in pairs:
        env.changeScenario(empty_sid)
        Z_empty = np.array(env.Z_filled, dtype=np.float32)

        env.changeScenario(occ_sid)
        Z_occ = np.array(env.Z_filled, dtype=np.float32)

        kernel_static = kernels_static[empty_sid]

        for run in range(int(n_runs)):
            seed = 1000 * run + int(occ_sid)

            pattern_std = generate_measurement_pattern(env, occ_sid, N_max=max(N_list),
                                                       strategy=pattern_strategy, seed=seed)

            pattern_dream = generate_measurement_pattern_dream(
                env, vae_base, dyn_base,
                empty_sid=empty_sid, occ_sid=occ_sid,
                N_max=max(N_list),
                seed=seed,
                **dream_cfg
            )

            for N in N_list:
                coords_std = pattern_std[:N]
                coords_dream = pattern_dream[:N]

                # copy-empty
                rmse_base = float(np.sqrt(np.mean((Z_empty - Z_occ) ** 2)))
                mae_base = float(np.mean(np.abs(Z_empty - Z_occ)))
                metrics["copy-empty"]["rmse"][N].append(rmse_base)
                metrics["copy-empty"]["mae"][N].append(mae_base)

                # GP
                rmse_gp, mae_gp, _ = gp_predict_occupied(env, occ_sid, coords_std, kernel_static)
                metrics["gp"]["rmse"][N].append(rmse_gp)
                metrics["gp"]["mae"][N].append(mae_gp)

                # worldmodel V on standard coords
                v_std = clone_vae(vae_base)
                if do_finetune and N > 0:
                    finetune_vae_on_coords(env, v_std, empty_sid, occ_sid, coords_std,
                                           steps_ft=steps_ft, lr_ft=lr_ft)
                rmse_wm, mae_wm, _ = vae_predict_occupied(env, v_std, empty_sid, occ_sid, coords_std)
                metrics["worldmodel"]["rmse"][N].append(rmse_wm)
                metrics["worldmodel"]["mae"][N].append(mae_wm)

                # worldmodel-dream on dream coords
                v_dw = clone_vae(vae_base)
                if do_finetune and N > 0:
                    finetune_vae_on_coords(env, v_dw, empty_sid, occ_sid, coords_dream,
                                           steps_ft=steps_ft, lr_ft=lr_ft)
                rmse_dw, mae_dw, _ = vae_predict_occupied(env, v_dw, empty_sid, occ_sid, coords_dream)
                metrics["worldmodel-dream"]["rmse"][N].append(rmse_dw)
                metrics["worldmodel-dream"]["mae"][N].append(mae_dw)

    print(f"\n=== RMSE (mean ± std), scale={SCALE_FACTOR} ===")
    header = "N | " + " | ".join([f"{m:>16s}" for m in methods])
    print(header)
    print("-" * len(header))
    for N in N_list:
        row = []
        for m in methods:
            arr = np.array(metrics[m]["rmse"][N], dtype=np.float32)
            row.append("n/a" if arr.size == 0 else f"{arr.mean():.2f}±{arr.std():.2f}")
        print(f"{N:2d} | " + " | ".join([f"{x:>16s}" for x in row]))

    return metrics


#  Plotting + saving images


def plot_rmse_curves(metrics, N_list, scale_factor, out_path):
    methods = list(metrics.keys())
    plt.figure(figsize=(8, 5))
    for m in methods:
        means = []
        for N in N_list:
            arr = np.array(metrics[m]["rmse"][N], dtype=np.float32)
            means.append(np.nan if arr.size == 0 else float(arr.mean()))
        plt.plot(N_list, means, marker="o", label=m)
    plt.xlabel("Number of measurements N")
    plt.ylabel("RMSE (RSSI units)")
    plt.title(f"RMSE vs N (scale={scale_factor})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved: {out_path}")


def plot_heatmaps_for_pair(env: SITEnvironment,
                           vae_base: VisionVAE,
                           dyn_base: LatentDynamics,
                           kernel_static,
                           empty_sid: int,
                           occ_sid: int,
                           N: int,
                           pattern_strategy: str = "farthest",
                           seed: int = 0,
                           steps_ft: int = 80,
                           lr_ft: float = 5e-4,
                           do_finetune=True,
                           dream_cfg=None,
                           out_path=None):
    if dream_cfg is None:
        dream_cfg = dict(candidate_pool=40, dream_samples=12)

    env.changeScenario(empty_sid)
    Z_empty = np.array(env.Z_filled, dtype=np.float32)

    env.changeScenario(occ_sid)
    Z_occ = np.array(env.Z_filled, dtype=np.float32)
    H, W = Z_occ.shape

    coords_std = generate_measurement_pattern(env, occ_sid, N_max=N, strategy=pattern_strategy, seed=seed)[:N]
    coords_dream = generate_measurement_pattern_dream(env, vae_base, dyn_base, empty_sid, occ_sid, N_max=N,
                                                      seed=seed, **dream_cfg)[:N]

    rmse_gp, _, Z_gp = gp_predict_occupied(env, occ_sid, coords_std, kernel_static)

    v_std = clone_vae(vae_base)
    if do_finetune and N > 0:
        finetune_vae_on_coords(env, v_std, empty_sid, occ_sid, coords_std,
                               steps_ft=steps_ft, lr_ft=lr_ft)
    rmse_wm, _, Z_wm = vae_predict_occupied(env, v_std, empty_sid, occ_sid, coords_std)

    v_dw = clone_vae(vae_base)
    if do_finetune and N > 0:
        finetune_vae_on_coords(env, v_dw, empty_sid, occ_sid, coords_dream,
                               steps_ft=steps_ft, lr_ft=lr_ft)
    rmse_dw, _, Z_dw = vae_predict_occupied(env, v_dw, empty_sid, occ_sid, coords_dream)

    rmse_base = float(np.sqrt(np.mean((Z_empty - Z_occ) ** 2)))

    vmin = min(Z_occ.min(), Z_wm.min(), Z_dw.min(), Z_gp.min(), Z_empty.min())
    vmax = max(Z_occ.max(), Z_wm.max(), Z_dw.max(), Z_gp.max(), Z_empty.max())

    def overlay(ax, coords, color="w"):
        if not coords:
            return
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        ax.scatter(xs, ys, s=max(8, 60 / max(1, SCALE_FACTOR)), c=color, marker="x", linewidths=1.0)

    plt.figure(figsize=(14, 8))

    ax = plt.subplot(2, 3, 1)
    ax.imshow(Z_empty, origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046)
    ax.set_title(f"copy-empty\nRMSE={rmse_base:.2f}")
    ax.axis("off")

    ax = plt.subplot(2, 3, 2)
    ax.imshow(Z_gp, origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046)
    ax.set_title(f"GP (std)\nRMSE={rmse_gp:.2f}")
    overlay(ax, coords_std)
    ax.axis("off")

    ax = plt.subplot(2, 3, 3)
    ax.imshow(Z_occ, origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046)
    ax.set_title("GT occupied")
    ax.axis("off")

    ax = plt.subplot(2, 3, 4)
    ax.imshow(Z_wm, origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046)
    ax.set_title(f"WorldModel V (std)\nRMSE={rmse_wm:.2f}")
    overlay(ax, coords_std)
    ax.axis("off")

    ax = plt.subplot(2, 3, 5)
    ax.imshow(Z_dw, origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046)
    ax.set_title(f"WorldModel (dream)\nRMSE={rmse_dw:.2f}")
    overlay(ax, coords_dream)
    ax.axis("off")

    ax = plt.subplot(2, 3, 6)
    m_std = np.zeros((H, W), dtype=np.float32)
    m_dw = np.zeros((H, W), dtype=np.float32)
    for (x, y) in coords_std:
        m_std[y, x] = 1.0
    for (x, y) in coords_dream:
        m_dw[y, x] = 1.0
    ax.imshow(m_std - m_dw, origin="lower", vmin=-1, vmax=1)
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046)
    ax.set_title("mask diff (std - dream)")
    ax.axis("off")

    plt.suptitle(f"Heatmaps (scale={SCALE_FACTOR}, empty={empty_sid}, occ={occ_sid}, N={N})", y=0.99)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[Plot] Saved: {out_path}")
    else:
        plt.show()


#  Main loop

if __name__ == "__main__":
    time_start = time.time()
    print("World Models-style (VAE+Dynamics) vs GP + Dreaming: multi-scale few-shot experiment")

    h5_path = "data/WiFi_RSSI_data2025-06-25_11-32-25.h5"

    overlapping_pairs = [(0, 12), (3, 13), (5, 14), (4, 15)]
    train_pairs = [(0, 12), (5, 14), (4, 15)]
    eval_pairs = [(3, 13)]

    N_list = [0, 1, 3, 5, 10, 15, 20, 25, 30, 35]
    scale_factors = [1, 2, 4, 8, 16]

    dream_cfg = dict(
        candidate_pool=40,
        dream_samples=12
    )

    for sf in scale_factors:
        SCALE_FACTOR = int(sf)
        print("\n" + "=" * 70)
        print(f"Scale factor = {SCALE_FACTOR}")
        print("=" * 70)

        env = SITEnvironment(scenario_id=0, h5_path=h5_path)

        env.changeScenario(overlapping_pairs[0][0])
        H, W = env.Z_filled.shape
        print(f"[Scale={SCALE_FACTOR}] Grid: H={H}, W={W}")

        # Global normalization (per scale)
        all_maps = []
        for (empty_sid, occ_sid) in overlapping_pairs:
            env.changeScenario(empty_sid)
            all_maps.append(np.array(env.Z_filled, dtype=np.float32))
            env.changeScenario(occ_sid)
            all_maps.append(np.array(env.Z_filled, dtype=np.float32))
        all_maps = np.stack(all_maps, axis=0)
        GLOBAL_MEAN = float(all_maps.mean())
        GLOBAL_STD = float(all_maps.std() + 1e-8)
        print(f"[Scale={SCALE_FACTOR}] GLOBAL_MEAN={GLOBAL_MEAN:.2f}, GLOBAL_STD={GLOBAL_STD:.2f}")

        # Build World Models components
        vae = VisionVAE(H, W, z_dim=64, base_channels=32)
        dyn = LatentDynamics(z_dim=64, hidden_dim=128, act_dim=2)

        train_kwargs = dict(
            env=env,
            train_pairs=train_pairs,
            N_choices=(0, 1, 3, 5, 10, 15, 20, 25),
            epochs=200,
            episodes_per_epoch=50,
            pattern_strategy="random",
            lr=1e-3,
            beta_kl=1e-3,
            seed=0
        )

        load_or_train_two_models(SCALE_FACTOR, vae, dyn, train_world_models, train_kwargs)

        # Fit GP kernels (per scale)
        static_ids = sorted(set([p[0] for p in overlapping_pairs]))
        kernels_static = {}
        for sid in static_ids:
            kernels_static[sid] = fit_gp_kernel_on_static(env, static_sid=sid, target_points=500, random_state=0)

        # Evaluate
        metrics = evaluate_worldmodel_vs_gp(
            env,
            vae_base=vae,
            dyn_base=dyn,
            kernels_static=kernels_static,
            pairs=eval_pairs,
            N_list=N_list,
            pattern_strategy="farthest",
            n_runs=5,
            steps_ft=80,
            lr_ft=5e-4,
            do_finetune=True,
            dream_cfg=dream_cfg
        )

        # CSV summary
        methods = ["copy-empty", "gp", "worldmodel", "worldmodel-dream"]
        rows = summarize_metrics_to_rows(metrics, methods, N_list)
        csv_path = os.path.join(RESULTS_DIR, f"summary_sf{SCALE_FACTOR:02d}.csv")
        write_summary_csv(rows, csv_path)

        # RMSE curve plot
        rmse_plot_path = os.path.join(PLOTS_DIR, f"rmse_curve_sf{SCALE_FACTOR:02d}.png")
        plot_rmse_curves(metrics, N_list, SCALE_FACTOR, rmse_plot_path)

        # Heatmap figure
        N_vis = 20
        for (empty_sid, occ_sid) in eval_pairs:
            heat_path = os.path.join(PLOTS_DIR, f"heatmaps_sf{SCALE_FACTOR:02d}_e{empty_sid}_o{occ_sid}_N{N_vis}.png")
            plot_heatmaps_for_pair(
                env,
                vae_base=vae,
                dyn_base=dyn,
                kernel_static=kernels_static[empty_sid],
                empty_sid=empty_sid,
                occ_sid=occ_sid,
                N=N_vis,
                pattern_strategy="farthest",
                seed=0,
                steps_ft=80,
                lr_ft=5e-4,
                do_finetune=True,
                dream_cfg=dream_cfg,
                out_path=heat_path
            )

    print(f"\nTotal experiment time: {time.time() - time_start:.1f} s")
