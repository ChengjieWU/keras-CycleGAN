import keras
from keras.layers import Conv2D, Activation, Input, Dropout, \
    Conv2DTranspose, LeakyReLU
from keras.layers.normalization import BatchNormalization
import numpy as np

from .network.layers import ReflectionPadding2D, InstanceNormalization
from .network.models import Xception
from .network.models import MobileNetV2


class CycleGAN:
    """This class implements CycleGAN, a type of GAN that can transfer images
    from domain A to domain B and vice verse.

        It consists of 4 models: g_a (transfer image from domain B to domain A),
    g_b, d_a(judge whether an image belongs to domain A), d_b.

        Typically, use compile, fit_generator and predict_generator three
    methods. Use load_weights and save to load or save weights.
    """

    def __init__(self, dis_mode="xception"):
        """CycleGAN.

        :param dis_mode: Discriminator mode, chosen from xception, patchgan and
            mobile currently.
        """
        self.dis_mode = dis_mode

        self.fake_a_pool = ImagePool(pool_size=0)
        self.fake_b_pool = ImagePool(pool_size=0)
        self.g_a = resnet_generator(input_nc=3, output_nc=3,
                                    ngf=64, norm_layer="InstanceNorm",
                                    use_dropout=True, n_blocks=9,
                                    padding_type="reflect")
        self.g_b = resnet_generator(input_nc=3, output_nc=3,
                                    ngf=64, norm_layer="InstanceNorm",
                                    use_dropout=True, n_blocks=9,
                                    padding_type="reflect")
        if dis_mode == "xception":
            self.d_a = xception_discriminator(input_nc=3, leaky_relu=0.2)
            self.d_b = xception_discriminator(input_nc=3, leaky_relu=0.2)
            self.dis_output_shape = (1,)
            self.gan_loss_mode = "bce"
        elif dis_mode == "mobile":
            self.d_a = mobilenet_discriminator(input_nc=3, leaky_relu=0.2)
            self.d_b = mobilenet_discriminator(input_nc=3, leaky_relu=0.2)
            self.dis_output_shape = (1,)
            self.gan_loss_mode = "bce"
        elif dis_mode == "patchgan":
            self.d_a = patchgan_discriminator(input_nc=3, ndf=64, n_layers=3,
                                              norm_layer="InstanceNorm")
            self.d_b = patchgan_discriminator(input_nc=3, ndf=64, n_layers=3,
                                              norm_layer="InstanceNorm")
            self.dis_output_shape = (28, 28, 512)
            self.gan_loss_mode = "lsgan"
        else:
            raise ValueError("%s discriminator is not implemented")
        self.training_g = self.build_training_g()

    def load_weights(self, weight_dict: dict):
        """Load pre-trained weights.

        :param weight_dict: A dictionary of model, weight.
        :return: None.
        """
        for m, w in weight_dict.items():
            if m == "g_a":
                self.g_a.load_weights(w)
            elif m == "g_b":
                self.g_b.load_weights(w)
            elif m == "d_a":
                self.d_a.load_weights(w)
            elif m == "d_b":
                self.d_b.load_weights(w)
            else:
                raise ValueError("No model named %s" % m)

    def save(self, weight_dict: dict):
        """Save model weights.

        :param weight_dict: A dictionary of model, path.
        :return: None.
        """
        for m, w in weight_dict.items():
            if m == "g_a":
                self.g_a.save(w)
            elif m == "g_b":
                self.g_b.save(w)
            elif m == "d_a":
                self.d_a.save(w)
            elif m == "d_b":
                self.d_b.save(w)
            else:
                raise ValueError("No model named %s" % m)

    def _get_gan_loss(self):
        """Get GAN loss using self.gan_loss_mode.

        This function should be called internally under most circumstances.

        :return: A keras loss instance.
        """
        if self.gan_loss_mode == "lsgan":
            gan_loss = keras.losses.mean_squared_error
        elif self.gan_loss_mode == "bce":
            gan_loss = keras.losses.binary_crossentropy
        elif self.gan_loss_mode == "vanilla":
            raise NotImplementedError("BCEwithLogits loss not implemented")
        elif self.gan_loss_mode == "wgangp":
            raise NotImplementedError("None loss not implemented")
        else:
            raise NotImplementedError(
                "Gan loss mode %s not implemented" % self.gan_loss_mode)
        return gan_loss

    def build_training_g(self):
        """Build generator ready for training.

        :return: A keras model ready for training.
        """
        input_a = keras.layers.Input(shape=(None, None, 3))
        input_b = keras.layers.Input(shape=(None, None, 3))
        fake_a = self.g_b(input_a)
        fake_b = self.g_a(input_b)
        rec_a = self.g_a(fake_a)
        rec_b = self.g_b(fake_b)
        idt_a = self.g_a(input_a)
        idt_b = self.g_b(input_b)
        deceive_a = self.d_b(fake_a)
        deceive_b = self.d_a(fake_b)
        deceive = keras.layers.Concatenate(axis=1)([deceive_a, deceive_b])
        training_g = keras.models.Model(
            inputs=[input_a, input_b],
            outputs=[idt_a, idt_b, rec_a, rec_b, deceive]
        )
        return training_g

    def compile_training_g(self, opt=None, lr=3e-4,
                           lambda_a=10.0, lambda_b=10.0,
                           lambda_identity=0.5, lambda_gan=1.0):
        """Compile self.training_g with given optimization parameters.

        :param opt: Optimizer instance. If none, Adam is used with learning rate
            specified by lr.
        :param lr: Learning rate. Only works when opt=None.
        :param lambda_a: Loss hyper-parameter.
        :param lambda_b: Loss hyper-parameter.
        :param lambda_identity: Loss hyper-parameter.
        :param lambda_gan: Loss hyper-parameter.
        :return: None.
        """
        gan_loss = self._get_gan_loss()
        self.d_a.trainable = False
        self.d_b.trainable = False
        if opt is None:
            opt = keras.optimizers.Adam(lr=lr)
        self.training_g.compile(
            optimizer=opt,
            loss=[keras.losses.mean_absolute_error,
                  keras.losses.mean_absolute_error,
                  keras.losses.mean_absolute_error,
                  keras.losses.mean_absolute_error,
                  gan_loss],
            loss_weights=[lambda_a * lambda_identity,
                          lambda_b * lambda_identity,
                          lambda_a, lambda_b, lambda_gan]
        )

    def compile_training_d(self, d, opt=None, lr=3e-4):
        """Build discriminator with given optimization parameters.

        :param d: The discriminator model.
        :param opt: Optimizer instance. If none, Adam is used with learning rate
            specified by lr.
        :param lr: Learning rate. Only works when opt=None.
        :return: A compiled keras model ready for training.
        """
        gan_loss = self._get_gan_loss()
        d.trainable = True
        if opt is None:
            opt = keras.optimizers.Adam(lr=lr)
        d.compile(optimizer=opt, loss=gan_loss)
        return d

    def compile(self, gen_opt=None, dis_opt=None, gen_lr=3e-4, dis_lr=3e-4,
                lambda_a=10.0, lambda_b=10.0,
                lambda_identity=0.5, lambda_gan=1.0):
        """Compile generators and discriminators for training.

        :param gen_opt: Optimizer instance for generator. If None, Adam is used
            with learning rate specified by gen_lr.
        :param dis_opt: Optimizer instance for discriminator. If None, Adam is
            used with learning rate specified by dis_lr.
        :param gen_lr: Learning rate for generator. Only works when
            gen_opt=None.
        :param dis_lr: Learning rate for discriminator. Only works when
            dis_opt=None.
        :param lambda_a: Loss hyper-parameter.
        :param lambda_b: Loss hyper-parameter.
        :param lambda_identity: Loss hyper-parameter.
        :param lambda_gan: Loss hyper-parameter.
        :return: None.
        """
        self.d_a = self.compile_training_d(self.d_a, opt=dis_opt, lr=dis_lr)
        self.d_b = self.compile_training_d(self.d_b, opt=dis_opt, lr=dis_lr)
        d_a_trainable = len(self.d_a.trainable_weights)
        d_b_trainable = len(self.d_b.trainable_weights)
        g_a_trainable = len(self.g_a.trainable_weights)
        g_b_trainable = len(self.g_b.trainable_weights)
        self.compile_training_g(opt=gen_opt,
                                lr=gen_lr,
                                lambda_a=lambda_a,
                                lambda_b=lambda_b,
                                lambda_identity=lambda_identity,
                                lambda_gan=lambda_gan)
        # These 3 assertions ensure that Keras is doing what GANs are
        # intended to do.
        assert len(self.d_a._collected_trainable_weights) == d_a_trainable
        assert len(self.d_b._collected_trainable_weights) == d_b_trainable
        assert len(self.training_g._collected_trainable_weights) == (
                g_a_trainable + g_b_trainable)

    def train_step_g(self, batch_a, batch_b):
        """Training step of generator.

        :param batch_a: A batch of data from domain A.
        :param batch_b: A batch of data from domain B.
        :return: Losses.
        """
        logs = self.training_g.train_on_batch(
            x=[batch_a, batch_b],
            y=[
                batch_a, batch_b, batch_a, batch_b,
                np.concatenate(
                    (np.zeros((batch_a.shape[0],) + self.dis_output_shape),
                     np.ones((batch_b.shape[0],) + self.dis_output_shape)),
                    axis=1)
            ]
        )
        return logs

    def train_step_d_a(self, batch_a, batch_b, pure=False):
        """Train step of discriminator of domain A.

        :param batch_a: A batch of domain A data.
        :param batch_b: A batch of domain B data.
        :param pure: Boolean, whether to train with fake data.
        :return: Losses, a list of length 2.
        """

        # NOTE! Because a bug in Keras, data passed to train_on_batch should
        # not belong to the same class. Therefore, the batch size here is
        # actually doubled.

        logs = list()
        fake_b = self.g_a.predict(batch_b)
        fake_b = self.fake_b_pool.query(fake_b)
        logs.append(self.d_a.train_on_batch(
            x=np.concatenate((batch_a, batch_b), axis=0),
            y=np.concatenate(
                (np.ones((batch_a.shape[0],) + self.dis_output_shape),
                 np.zeros((batch_b.shape[0],) + self.dis_output_shape)), axis=0)
        ))
        if not pure:
            logs.append(self.d_a.train_on_batch(
                x=np.concatenate((batch_a, fake_b), axis=0),
                y=np.concatenate(
                    (np.ones((batch_a.shape[0],) + self.dis_output_shape),
                     np.zeros((fake_b.shape[0],) + self.dis_output_shape)),
                    axis=0)
            ))
        else:
            logs.append(0.)
        return logs

    def train_step_d_b(self, batch_a, batch_b, pure=False):
        """Train step of discriminator of domain B.

        :param batch_a: A batch of domain A data.
        :param batch_b: A batch of domain B data.
        :param pure: Boolean, whether to train with fake data.
        :return: Losses, a list of length 2.
        """

        # NOTE! Because a bug in Keras, data passed to train_on_batch should
        # not belong to the same class. Therefore, the batch size here is
        # actually doubled.

        logs = list()
        fake_a = self.g_b.predict(batch_a)
        fake_a = self.fake_a_pool.query(fake_a)
        logs.append(self.d_b.train_on_batch(
            x=np.concatenate((batch_a, batch_b), axis=0),
            y=np.concatenate(
                (np.ones((batch_a.shape[0],) + self.dis_output_shape),
                 np.zeros((batch_b.shape[0],) + self.dis_output_shape)), axis=0)
        ))
        if not pure:
            logs.append(self.d_b.train_on_batch(
                x=np.concatenate((fake_a, batch_b)),
                y=np.concatenate(
                    (np.ones((fake_a.shape[0],) + self.dis_output_shape),
                     np.zeros((batch_b.shape[0],) + self.dis_output_shape)),
                    axis=0)
            ))
        else:
            logs.append(0.)
        return logs

    def train_step(self, data_a, data_b, batch_size=4, mode="both"):
        """Train step of the whole CycleGAN model.

        :param data_a: A generator of data from domain A.
        :param data_b: A generator of data from domain B.
        :param batch_size: Int, batch size.
        :param mode: One of "both", "gen", "dis" and "dis-pure". Models to
            train.
        :return: A dictionary of losses.
        """
        if mode not in ["both", "gen", "dis", "dis-pure"]:
            raise ValueError("Mode name %s is not supported" % mode)

        logs = {"gen": None, "d_a": None, "d_b": None}

        def _get_batch_data(gen):
            data = next(gen)[0]
            if data.shape[0] != batch_size:
                data = next(gen)[0]
            return data

        batch_a = _get_batch_data(data_a)
        batch_b = _get_batch_data(data_b)
        if "dis" not in mode:
            logs["gen"] = self.train_step_g(batch_a, batch_b)

        if "gen" not in mode:
            logs["d_a"] = self.train_step_d_a(batch_a, batch_b,
                                              mode == "dis-pure")
            logs["d_b"] = self.train_step_d_b(batch_a, batch_b,
                                              mode == "dis-pure")

        return logs

    def predict_generator(self, m, *args, **kwargs):
        """Predict using data from a generator.

        The predict_generator method of keras models are wrapped.

        :param m: Model name.
        :param args: Arguments for predict_generator.
        :param kwargs: Arguments for predict_generator.
        :return: None.
        """
        if m == "d_a":
            return self.d_a.predict_generator(*args, **kwargs)
        elif m == "d_b":
            return self.d_b.predict_generator(*args, **kwargs)
        elif m == "g_a":
            return self.g_a.predict_generator(*args, **kwargs)
        elif m == "g_b":
            return self.g_b.predict_generator(*args, **kwargs)
        else:
            raise ValueError("No model named %s" % m)

    def predict(self, m, *args, **kwargs):
        """Predict using data from a numpy array.

        The predict method of keras models are wrapped.

        :param m: Model name.
        :param args: Arguments for predict.
        :param kwargs: Arguments for predict.
        :return: None.
        """
        if m == "d_a":
            return self.d_a.predict(*args, **kwargs)
        elif m == "d_b":
            return self.d_b.predict(*args, **kwargs)
        elif m == "g_a":
            return self.g_a.predict(*args, **kwargs)
        elif m == "g_b":
            return self.g_b.predict(*args, **kwargs)
        else:
            raise ValueError("No model named %s" % m)

    def fit_generator(self, data_a, data_b, steps_per_epoch, batch_size=4,
                      mode="both", epochs=1, verbose=0):
        """Train CycleGAN model.

        :param data_a: Data generator for domain A.
        :param data_b: Data generator for domain B.
        :param steps_per_epoch: Number of steps per epoch. After an epoch, loss
            statistics are printed.
        :param batch_size: Batch size.
        :param mode: One of "both", "gen", "dis" and "dis-pure". Models to
            train.
        :param epochs: Number of epochs.
        :param verbose: Whether to print inner logs.
        :return: None.
        """
        if mode not in ["both", "gen", "dis", "dis-pure"]:
            raise ValueError("Mode name %s is not supported" % mode)

        for epoch in range(epochs):
            print("Epoch %d/%d" % (epoch+1, epochs))
            avg_logs = {
                "gen": np.zeros(6), "d_a": np.zeros(2), "d_b": np.zeros(2)}
            for step in range(steps_per_epoch):
                logs = self.train_step(data_a, data_b, batch_size, mode)
                if verbose:
                    print(logs)
                if "dis" not in mode:
                    avg_logs["gen"] = avg_logs["gen"] * step / (step+1) \
                        + np.array(logs["gen"]) / (step+1)
                if "gen" not in mode:
                    avg_logs["d_a"] = avg_logs["d_a"] * step / (step + 1) \
                        + np.array(logs["d_a"]) / (step + 1)
                    avg_logs["d_b"] = avg_logs["d_b"] * step / (step + 1) \
                        + np.array(logs["d_b"]) / (step + 1)
            if "dis" not in mode:
                print("total loss: %f, idt_a: %f, idt_b: %f, rec_a: %f, rec_b: "
                      "%f, gan_loss: %f." % tuple(avg_logs["gen"]))
            if "gen" not in mode:
                print("clf_loss: %f, fake_loss: %f" %
                      tuple(avg_logs["d_a"]))
                print("clf_loss: %f, fake_loss: %f" %
                      tuple(avg_logs["d_b"]))


def resnet_generator(input_nc, output_nc, ngf=64,
                     norm_layer="InstanceNorm",
                     use_dropout=False, n_blocks=6, padding_type="reflect"):
    """Create a ResNet geneartor.

    :param input_nc: Int, input channels.
    :param output_nc: Int, output channels.
    :param ngf: Number of filters in the last conv layer.
    :param norm_layer: Type of normalization layer. One of "InstanceNorm",
        "BatchNorm".
    :param use_dropout: Boolean, whether to use dropout in ResNet blocks.
    :param n_blocks: Number of ResNet blocks.
    :param padding_type: Padding type. One of "reflect", "replicate" and "zero".
    :return: A keras model.
    """

    if norm_layer == "InstanceNorm":
        norm_layer = InstanceNormalization
        use_bias = True
    elif norm_layer == "BatchNorm":
        norm_layer = BatchNormalization
        use_bias = False
    else:
        raise ValueError("Invalid normalization")

    inp = Input(shape=(None, None, input_nc))
    x = ReflectionPadding2D(padding=3)(inp)
    x = Conv2D(ngf, kernel_size=7, strides=1, padding="valid",
               use_bias=use_bias)(x)
    x = norm_layer(axis=3)(x)
    x = Activation("relu")(x)

    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2 ** i
        x = Conv2D(ngf * mult * 2, kernel_size=3, strides=2,
                   padding="same", use_bias=use_bias)(x)
        x = norm_layer(axis=3)(x)
        x = Activation("relu")(x)

    mult = 2 ** n_downsampling
    for i in range(n_blocks):
        x = resnet_block(x, ngf * mult, padding_type=padding_type,
                         norm_layer=norm_layer,
                         use_dropout=use_dropout,
                         use_bias=use_bias)

    for i in range(n_downsampling):
        mult = 2 ** (n_downsampling - i)
        x = Conv2DTranspose(int(ngf * mult / 2), kernel_size=3, strides=2,
                            padding="same", use_bias=use_bias)(x)
        x = norm_layer(axis=3)(x)
        x = Activation("relu")(x)

    x = ReflectionPadding2D(padding=3)(x)
    x = Conv2D(output_nc, kernel_size=7, strides=1, padding="valid")(x)
    x = Activation("tanh")(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    return model


def resnet_block(x, dim, padding_type, norm_layer, use_dropout, use_bias):
    """Create a ResNet block.

    :param x: Input tensor.
    :param dim: Number of channels.
    :param padding_type: Padding type. One of "reflect", "replicate" and "zero".
    :param norm_layer: Type of normalization layer. One of "InstanceNorm",
        "BatchNorm".
    :param use_dropout: Boolean, whether to use dropout.
    :param use_bias: Boolean, whether to use bias.
    :return: Tensor after a ResNet block.
    """
    inp = x
    if padding_type == "reflect":
        x = ReflectionPadding2D(1)(inp)
        x = Conv2D(dim, kernel_size=3, strides=1,
                   padding="valid", use_bias=use_bias)(x)
    elif padding_type == "replicate":
        # TODO: implement replicate padding in keras
        raise NotImplementedError(
            "Replicate padding is not implemented yet!")
    elif padding_type == "zero":
        x = Conv2D(dim, kernel_size=3, strides=1,
                   padding="same", use_bias=use_bias)(x)
    else:
        raise ValueError("Invalid padding type '%s'" % padding_type)

    x = norm_layer(axis=3)(x)
    x = Activation("relu")(x)

    if use_dropout:
        x = Dropout(0.5)(x)

    if padding_type == "reflect":
        x = ReflectionPadding2D(1)(x)
        x = Conv2D(dim, kernel_size=3, strides=1,
                   padding="valid", use_bias=use_bias)(x)
    elif padding_type == "replicate":
        # TODO: implement replicate padding in keras
        raise NotImplementedError(
            "Replicate padding is not implemented yet!")
    elif padding_type == "zero":
        x = Conv2D(dim, kernel_size=3, strides=1,
                   padding="same", use_bias=use_bias)(x)

    x = norm_layer(axis=3)(x)
    x = keras.layers.Add()([x, inp])

    return x


def patchgan_discriminator(input_nc, ndf=64, n_layers=3,
                           norm_layer="InstanceNorm"):
    """Create a PatchGAN discriminator.

    :param input_nc: Int, input channels.
    :param ndf: Number of filters in the last conv layer.
    :param n_layers: Int, number of PatchGAN layers.
    :param norm_layer: Type of normalization layer. One of "InstanceNorm",
        "BatchNorm".
    :return: A keras model.
    """
    if norm_layer == "InstanceNorm":
        norm_layer = InstanceNormalization
        use_bias = True
    elif norm_layer == "BatchNorm":
        norm_layer = BatchNormalization
        use_bias = False
    else:
        raise ValueError("Invalid normalization")
    kw = 4

    inp = Input(shape=(None, None, input_nc))
    x = Conv2D(ndf, kernel_size=kw, strides=2, padding="same")(inp)
    x = LeakyReLU(0.2)(x)

    for n in range(1, n_layers):
        nf_mult = min(2**n, 8)
        x = Conv2D(ndf * nf_mult, kernel_size=kw, strides=2, padding="same",
                   use_bias=use_bias)(x)
        x = norm_layer(axis=3)(x)
        x = LeakyReLU(0.2)(x)

    nf_mult = min(2 ** n_layers, 8)
    x = Conv2D(ndf * nf_mult, kernel_size=kw, strides=1, padding="same")(x)

    model = keras.models.Model(inputs=inp, outputs=x)

    return model


def xception_discriminator(input_nc, leaky_relu=None):
    """Create a Xception discriminator.

    :param input_nc: Int, input channels.
    :param leaky_relu: Float, used in LeakyReLU. If none, leaky_relu is not
        used.
    :return: A keras model.
    """
    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(None, None, input_nc),
        pooling="avg",
        leaky_relu=leaky_relu
    )
    seed = 12345
    x = base_model.output
    x = keras.layers.Dense(
        units=1, activation='sigmoid',
        use_bias=False,
        kernel_initializer=keras.initializers.glorot_uniform(seed),
        kernel_regularizer=keras.regularizers.l2(0.001)
    )(x)
    model = keras.models.Model(inputs=base_model.input, outputs=x)
    return model


def mobilenet_discriminator(input_nc, leaky_relu=None):
    """Create a MobileNetV2 discriminator.

    :param input_nc: Int, input channels.
    :param leaky_relu: Float, used in LeakyReLU. If none, leaky_relu is not
        used.
    :return: A keras model.
    """
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(None, None, input_nc),
        pooling="avg",
        leaky_relu=leaky_relu
    )
    seed = 12345
    x = base_model.output
    x = keras.layers.Dense(
        units=1, activation='sigmoid',
        use_bias=False,
        kernel_initializer=keras.initializers.glorot_uniform(seed),
        kernel_regularizer=keras.regularizers.l2(0.001)
    )(x)
    model = keras.models.Model(inputs=base_model.input, outputs=x)
    return model


class ImagePool:
    """This classes implements an image buffer that stores previously
    generated images.

    This buffer enables us to update discriminators using a history of
    generated images rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class.

        :param pool_size: int, the size of image buffer. If pool_size = 0,
            no buffer will be created.
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return images from the pool.

        Whenever the buffer is not full, new images are added and returned.
        Otherwise, by 50%, the buffer will return input images.
        By 50%, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.

        :param images: The latest generated images from the generator.
        :return: Images from the buffer.
        """
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = np.random.rand()
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size)
                    tmp = self.images[random_id].copy()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return np.array(return_images)


if __name__ == "__main__":
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    gpu_config = KTF.set_session(sess)

    """Prepare data here. Data should be a 4-d numpy array in two domains.
    
    Sample code:
    """
    a_train = np.random.rand(32, 224, 224, 3)
    a_train_labels = np.ones(32)
    b_train = np.random.rand(32, 224, 224, 3)
    b_train_labels = np.zeros(32)

    batch_size = 4

    datagen = keras.preprocessing.image.ImageDataGenerator()

    gen_a_train = datagen.flow(a_train, a_train_labels,
                               batch_size=batch_size, shuffle=True)
    gen_b_train = datagen.flow(b_train, b_train_labels,
                               batch_size=batch_size, shuffle=True)

    ############################################################
    #  Initiate a CycleGAN model.
    ############################################################
    model = CycleGAN(dis_mode="xception")

    ############################################################
    #  Train model.
    ############################################################
    dis_opt = keras.optimizers.Adam(lr=0.0002)
    gen_opt = keras.optimizers.Adam(lr=0.001)
    model.compile(gen_opt=gen_opt, dis_opt=dis_opt, lambda_a=10.0,
                  lambda_b=10.0,
                  lambda_identity=0.5, lambda_gan=0.0)

    model.fit_generator(gen_a_train, gen_b_train,
                        steps_per_epoch=20, batch_size=4,
                        mode="both", epochs=10)
