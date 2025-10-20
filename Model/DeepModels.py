import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import numpy as np


class DeepModels:
    """
    Derin öğrenme tabanlı anomali tespiti modelleri:
    Autoencoder, Variational Autoencoder (VAE), DeepSVDD.
    """

    @staticmethod
    def run_autoencoder(X, y_true=None, epochs=50, batch_size=32, learning_rate=0.001):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        input_dim = X_scaled.shape[1]

        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(16, activation='relu')(input_layer)
        encoded = layers.Dense(8, activation='relu')(encoded)
        decoded = layers.Dense(16, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)

        autoencoder = models.Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss='mse')

        autoencoder.fit(X_scaled, X_scaled,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0)

        X_pred = autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - X_pred), axis=1)

        # Threshold
        if y_true is not None and np.any(y_true == 1):
            n_anomalies = np.sum(y_true == 1)
            threshold = np.sort(mse)[-n_anomalies]
        else:
            threshold = np.percentile(mse, 95)

        y_pred = (mse > threshold).astype(int)
        return y_pred, autoencoder

    # ----------------------------------------------------- #
    @staticmethod
    def run_vae(X, y_true=None, epochs=15, batch_size=8, learning_rate=0.001, latent_dim=2):
        """
        SHAP uyumlu, kararlı ve küçük veri setlerinde anlamlı sonuçlar üreten
        Variational Autoencoder (VAE) modeli.
        - Deterministic sampling (eps=0)
        - Rekonstrüksiyon hatasına dayalı anomali skoru
        - Küçük latent_dim ve epoch: aşırı öğrenmeyi engeller
        """
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        # --- 1️⃣ Veri ölçekleme ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.astype(np.float32))
        input_dim = X_scaled.shape[1]

        # --- 2️⃣ Encoder ---
        inputs = layers.Input(shape=(input_dim,), name="vae_input")
        h = layers.Dense(16, activation='relu')(inputs)
        h = layers.Dense(8, activation='relu')(h)
        z_mean = layers.Dense(latent_dim, name="z_mean")(h)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)

        # Deterministik sampling (eps=0) → SHAP uyumlu
        def deterministic_sampling(args):
            z_mean_tensor, z_log_var_tensor = args
            return z_mean_tensor

        z = layers.Lambda(deterministic_sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

        # --- 3️⃣ Decoder (bağımsız tanımlı, SHAP kararlı) ---
        decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")
        h_dec = layers.Dense(8, activation='relu')(decoder_input)
        h_dec = layers.Dense(16, activation='relu')(h_dec)
        decoder_output = layers.Dense(input_dim, activation='linear', name="decoder_output")(h_dec)
        decoder = models.Model(decoder_input, decoder_output, name="decoder")

        # Decoder’ı bağla
        outputs = decoder(z)

        # --- 4️⃣ VAE modeli (training için) ---
        vae = models.Model(inputs, outputs, name="vae")

        # Reconstruction + KL loss
        reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = reconstruction_loss + kl_loss
        vae.add_loss(total_loss)
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        # --- 5️⃣ Modeli eğit ---
        vae.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

        # --- 6️⃣ Rekonstrüksiyon hatası üzerinden tahmin ---
        X_pred = vae.predict(X_scaled, batch_size=batch_size, verbose=0)
        mse = np.mean(np.square(X_scaled - X_pred), axis=1)

        # Dinamik eşik: %95 persentil veya gerçek etiket sayısına göre
        threshold = np.percentile(mse, 95)
        if y_true is not None and np.any(y_true == 1):
            n_anom = np.sum(y_true == 1)
            threshold = np.sort(mse)[-n_anom] if n_anom < len(mse) else threshold

        y_pred = (mse > threshold).astype(int)

        # --- 7️⃣ Modeli SHAP ile uyumlu hale getir ---
        vae.scaler = scaler  # orijinal ölçek referansı
        vae.decoder = decoder  # bağımsız decoder erişimi
        vae.latent_dim = latent_dim

        print(f"[VAE] Training complete. Threshold={threshold:.4f}, Mean MSE={mse.mean():.6f}")

        return y_pred, vae

    # ----------------------------------------------------- #
    @staticmethod
    @staticmethod
    def run_deepsvdd(X, y_true=None, params=None):
        """
        DeepSVDD modeli — SHAP uyumlu, dictionary tabanlı parametre girişi.
        Args:
            X: Veri matrisi
            y_true: Gerçek etiketler (opsiyonel)
            params: {"epochs": 50, "learning_rate": 0.001, "hidden_units": [32, 16, 8]}
        Returns:
            y_pred: Tahmin edilen etiketler (0/1)
            model: Eğitilmiş TF modeli (model.c attribute içerir)
        """
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        # --- Parametreleri dictionary'den al ---
        if params is None:
            params = {}
        epochs = params.get("epochs", 50)
        learning_rate = params.get("learning_rate", 0.001)
        hidden_units = params.get("hidden_units", [32, 16, 8])

        print(f"[DeepSVDD] Params -> epochs={epochs}, lr={learning_rate}, layers={hidden_units}")

        # --- 1️⃣ Veri ölçekleme ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.astype(np.float32))
        input_dim = X_scaled.shape[1]

        # --- 2️⃣ Model tanımı ---
        inputs = layers.Input(shape=(input_dim,), name="input")
        h = inputs
        for units in hidden_units:
            h = layers.Dense(units, activation='relu')(h)
        z = layers.Dense(hidden_units[-1], activation=None, name="embedding")(h)

        model = models.Model(inputs, z, name="DeepSVDD")

        # --- 3️⃣ Merkez vektör (c) ---
        c = np.mean(model.predict(X_scaled, verbose=0), axis=0)
        model.c = c  # SHAP açıklaması için kaydet

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # --- 4️⃣ Eğitim döngüsü ---
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                z_out = model(X_scaled, training=True)
                dist = tf.reduce_sum((z_out - c) ** 2, axis=1)
                loss = tf.reduce_mean(dist)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # --- 5️⃣ Tahmin & Eşik belirleme ---
        z_out = model.predict(X_scaled, verbose=0)
        dist = np.sum((z_out - c) ** 2, axis=1)

        threshold = np.percentile(dist, 95)
        if y_true is not None and np.any(y_true == 1):
            n_anomalies = np.sum(y_true == 1)
            threshold = np.sort(dist)[-n_anomalies] if n_anomalies > 0 else threshold

        y_pred = (dist > threshold).astype(int)

        # --- 6️⃣ Ek bilgiler ---
        model.scaler = scaler
        print(f"[DeepSVDD] Training done. Threshold={threshold:.5f}, Mean Dist={dist.mean():.5f}")

        return y_pred, model


