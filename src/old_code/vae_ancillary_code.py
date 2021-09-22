class DenseVAE:
    """VAE for outlier detection using tf.keras"""

    ############################################################################

    def __init__(self, encoder: "keras.model", decoder: "keras.model"):

        self.encoder = encoder.encoder
        self.decoder = decoder.decoder

        self.n_input_dimensions = encoder.n_input_dimensions
        self.inputs = Input(
            shape=(self.n_input_dimensions,), name="vae_input_layer"
        )

        self.latent_mu = encoder.latent_mu
        self.latent_ln_sigma = encoder.latent_ln_sigma
        self.loss = None

        self.vae = self.build_vae()

    ############################################################################
    def fit(
        self,
        spectra: "2D np.array",
        batch_size: "int" = None,
        epochs: "int" = 1,
    ) -> "None":

        self.vae.fit(
            x=spectra, y=spectra, epochs=epochs, batch_size=batch_size
        )

    ############################################################################
    def predict(self, spectra: "2D np.array") -> "2D np.array":

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)

        return self.vae.predict(spectra)

    ############################################################################
    def encode(self, spectra: "2D np.array") -> "2D np.array":

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        return self.encoder(spectra)

    ############################################################################
    def decode(self, coding: "2D np.array") -> "2D np.aray":

        if coding.ndim == 1:
            coding = coding.reshape(1, -1)

        return self.decoder(coding)

    ############################################################################
    def save_vae(self, fname: "str" = "DenseVAE"):

        self.vae.save(f"{fname}")

    ############################################################################
    def save_encoder(self, fname: "str" = "DenseEncoder"):

        self.encoder.save(f"{fname}")

    ############################################################################
    def save_decoder(self, fname: "str" = "DenseDecoder"):

        self.decoder.save(f"{fname}")

    ############################################################################
    def plot_model(self):

        plot_model(self.vae, to_file="DenseVAE.png", show_shapes="True")
        plot_model(
            self.encoder, to_file="DenseEncoder.png", show_shapes="True"
        )
        plot_model(
            self.decoder, to_file="DenseDecoder.png", show_shapes="True"
        )

    ############################################################################

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.vae.summary()

    ############################################################################

    def build_vae(self):

        vae = Model(
            self.inputs,
            self.decoder(self.encoder(self.inputs)),
            name="DenseVAE",
        )

        self.loss = self.vae_loss()
        vae.compile(loss=self.loss, optimizer="adam")  # , lr = self.lr)

        return vae

    ############################################################################

    def vae_loss(self):
        return self.vae_loss_aux

    ############################################################################

    def vae_loss_aux(self, y_true, y_pred):

        kl_loss = self.kl_loss()
        rec_loss = self.rec_loss(y_true, y_pred)
        return K.mean(kl_loss + rec_loss)

    ############################################################################
    def kl_loss(self):

        z_m = self.latent_mu
        z_s = self.latent_ln_sigma

        kl_loss = 1 + z_s - K.square(z_m) - K.exp(z_s)

        return -0.5 * K.sum(kl_loss, axis=-1)

    ############################################################################
    def rec_loss(self, y_true, y_pred):

        return keras.losses.mse(y_true, y_pred)
