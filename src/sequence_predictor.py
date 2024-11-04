import math
import torch
import torch.nn.functional as F
import lightning as L


class SoftAttention(L.LightningModule):
    def __init__(self, learning_rate=0.001, batch_size=10, unfreeze=0, random_text=False, random_everything=False,
                 fixed_text=False, random_images=False):
        super(SoftAttention, self).__init__()
        self.my_optimizer = None
        self.my_scheduler = None
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.frozen = False
        self.unfreeze_epoch = unfreeze
        self.loss_method = torch.nn.CrossEntropyLoss()
        self.train_sum_precision = 0
        self.train_sum_accuracy = 0
        self.train_sum_recall = 0
        self.train_sum_runs = 0
        self.val_sum_precision = 0
        self.val_sum_accuracy = 0
        self.val_sum_recall = 0
        self.val_sum_runs = 0


        self.text_reduction = torch.nn.Linear(512, 256)
        self.image_reduction = torch.nn.Linear(512, 256)

        self.W_query_text_half_dim = torch.nn.Linear(256, 256)
        self.W_query_image_half_dim = torch.nn.Linear(256, 256)
        self.W_query_text_full_dim = torch.nn.Linear(512, 512)
        self.W_query_image_full_dim = torch.nn.Linear(512, 512)

        self.W_key_text_half_dim = torch.nn.Linear(256, 256)
        self.W_key_image_half_dim = torch.nn.Linear(256, 256)
        self.W_key_image_full_dim = torch.nn.Linear(512, 512)
        self.W_key_text_full_dim = torch.nn.Linear(512, 512)

        self.fixed_text = torch.tensor([2.2875e-01, 2.3762e-02, 1.3448e-01, 6.5997e-02, 2.5605e-01,
                                        -1.6183e-01, 7.1169e-03, -1.6895e+00, 1.8110e-01, 1.7249e-01,
                                        7.0582e-02, -6.3566e-02, -1.5862e-01, -2.3586e-01, 6.9382e-02,
                                        9.4649e-02, 6.3127e-01, -4.1287e-02, -4.9883e-02, -2.1821e-01,
                                        5.8677e-01, -2.5353e-01, 1.4792e-01, 2.2195e-02, -6.8436e-02,
                                        -1.5512e-01, -9.8894e-02, 6.3377e-02, -2.3078e-01, 9.3588e-02,
                                        5.2875e-02, -5.1388e-01, -7.0461e-02, 2.4253e-02, -7.8069e-02,
                                        7.6921e-02, -1.1610e-01, -1.3345e-01, 7.8038e-03, -2.0226e-01,
                                        1.1381e-01, -9.6335e-02, -2.2195e-02, -6.5028e-02, 1.4025e-01,
                                        2.6969e-01, -1.0758e-01, 3.6736e-02, 3.2893e-01, -1.9067e-01,
                                        4.9070e-02, 8.0207e-02, 7.2942e-02, 7.7496e-03, 2.0883e-01,
                                        1.7339e-01, 1.0072e-01, -1.7874e-01, -4.6898e-02, -6.2682e-02,
                                        5.9596e-02, 5.2925e-02, 2.4633e-01, -7.2811e-02, -1.4157e-01,
                                        8.8013e-03, -4.6815e-02, -7.4260e-02, 8.6530e-03, -1.8174e-01,
                                        1.6101e-01, -4.8832e-02, -5.8030e-02, -3.2518e-02, -6.2896e-02,
                                        -2.3472e-01, -8.0996e-02, 1.1261e-01, -2.1039e-01, -2.3837e-01,
                                        -2.6827e-02, -2.3075e-01, -2.2087e-02, 5.4009e-01, 3.7671e-02,
                                        3.3140e-01, -4.2569e-02, -1.6946e-01, 1.7165e-01, 3.0887e-01,
                                        4.9847e-02, 1.2438e-02, -2.0701e+00, 2.7104e-01, 1.9001e-01,
                                        3.1907e-01, -9.1116e-02, -8.3141e-02, 4.5765e-03, -2.5675e-01,
                                        -2.2119e-02, 3.4949e-02, 2.8192e-01, 7.9688e-02, -2.1810e-01,
                                        8.1565e-02, 3.3208e-01, -9.1857e-02, -2.1145e-01, -1.6843e-01,
                                        6.7942e-02, 5.1067e-01, -1.6835e-01, 2.2090e-02, 1.8061e-02,
                                        -2.1313e-01, 2.6867e-02, -2.2734e-01, 8.4164e-02, -4.7868e-02,
                                        2.0980e-02, -2.1424e-01, -2.2919e-02, 1.7554e-01, 5.2253e-02,
                                        -2.2049e-01, 6.9408e-02, 7.0811e-02, -1.1892e-02, -4.7958e-02,
                                        7.9476e-02, 1.8851e-01, 2.2516e-02, 8.6119e+00, -7.8583e-02,
                                        1.0218e-01, 1.6675e-01, -4.0961e-01, 4.5291e-02, 7.9783e-02,
                                        -1.1764e-01, -2.3162e-01, -2.7717e-02, 1.2963e-01, -3.0165e-01,
                                        -2.1588e-02, -1.2324e-01, 1.9732e-02, -1.9312e-01, -7.1229e-02,
                                        2.5102e-01, -4.1674e-01, -1.5610e-01, -6.1321e-03, -4.5332e-02,
                                        6.1500e-02, -1.5942e-01, 3.5142e-01, -2.1119e-01, 4.5057e-02,
                                        -5.6277e-02, -3.4298e-01, -1.6499e-01, -2.9384e-02, -2.7163e-01,
                                        6.5339e-03, 2.7674e-02, -1.1302e-01, -2.6373e-02, -1.4370e-01,
                                        2.1936e-01, 1.3103e-01, 2.5538e-01, 1.9502e-01, -1.5278e-01,
                                        1.4978e-01, -2.5552e-01, 2.2397e-01, -1.0369e-01, -1.0491e-01,
                                        5.1112e-01, 2.4879e-01, 7.0940e-02, 1.7351e-01, -3.6831e-02,
                                        1.5027e-01, -1.9452e-01, 2.0322e-01, 8.5931e-02, -2.8588e-03,
                                        3.1146e-02, -3.3307e-01, 1.1595e-01, 1.9435e-01, -3.4536e-02,
                                        2.5245e-01, 4.5388e-02, 2.1197e-02, 4.2232e-02, 4.2436e-02,
                                        4.9622e-02, -2.0907e-01, 1.2264e-01, -7.3529e-02, -2.1788e-01,
                                        -1.2429e-01, -8.1422e-02, 1.6572e-01, -6.0989e-02, 8.0322e-02,
                                        3.3477e-01, -7.2207e-02, -8.8658e-02, -2.4944e-01, 9.9211e-02,
                                        8.6244e-02, 8.8807e-02, -1.9676e-01, -4.5365e-03, -3.7754e-01,
                                        -1.7204e-01, -1.3001e-01, 6.4961e-02, -5.8192e-03, 2.4670e-01,
                                        -8.3591e-02, -3.0810e-01, -3.4549e-02, -1.4452e-01, -5.5416e-02,
                                        1.0527e-02, 3.1159e-01, -1.3857e-01, -2.2676e-01, 1.4768e-01,
                                        3.2650e-01, 2.3971e-01, 6.8196e-02, -2.6235e-02, -2.9741e-01,
                                        4.7721e-02, -1.2859e-02, 2.0340e-01, 1.7823e-02, -1.1337e-01,
                                        4.4077e-02, -1.3949e-01, 2.9229e-01, 1.7425e-01, -5.0722e-03,
                                        -6.3722e-02, 1.0181e-01, 2.3344e-02, 2.2200e-01, 3.5022e-02,
                                        1.5361e-01, -1.0702e-03, 2.9319e-02, 1.8938e-01, -7.2263e-02,
                                        2.2192e-02, 9.5394e-02, -4.4459e-03, 7.6698e-02, -1.7830e-01,
                                        1.0213e-01, -8.8493e-02, -1.6439e-01, -1.1085e-01, 1.2938e-01,
                                        2.3929e-01, -4.9047e-02, -1.2814e-01, -2.1075e-01, 2.4423e-01,
                                        -4.4565e-02, -5.1225e-02, -4.0214e-02, -1.4033e-01, 6.3284e-02,
                                        4.7094e-01, -2.6821e-02, 2.1138e-02, 1.1590e-01, -2.0023e-02,
                                        1.7200e-01, 3.8215e-01, -2.4871e-01, -1.5359e-01, 2.4691e-01,
                                        1.4904e-01, -1.0636e-01, 2.4185e-01, 1.7119e-03, 1.4618e-01,
                                        -1.6813e-01, -4.4372e-01, -1.7475e-01, -6.9891e-02, -4.5553e-02,
                                        9.3102e-02, 1.7686e-02, -1.1781e-01, 6.9423e-02, 1.0211e-02,
                                        3.2742e-01, 7.5272e-02, 8.5080e-02, -1.7731e-01, 1.4030e-01,
                                        2.7764e-01, -6.5041e-02, 8.5968e+00, 2.5900e-01, -2.0825e-01,
                                        9.6241e-02, -1.5257e-01, -3.4269e-01, -1.1251e-01, 3.0549e-01,
                                        3.1628e-01, 6.1856e-01, 1.5791e-03, 6.5656e-02, 1.8862e-02,
                                        -7.1927e-02, 1.3239e-01, -1.1126e-01, 1.1135e-02, -3.2411e+00,
                                        -4.7349e-02, 1.4775e-01, -9.7712e-02, 4.5727e-02, -1.3868e-01,
                                        2.1260e-01, 1.5465e-01, 1.1308e-01, -8.0110e-02, -1.3123e-01,
                                        1.8527e-01, -8.6424e-02, -1.9778e-01, -1.3295e-01, -1.5880e-01,
                                        2.0800e-01, -3.6513e-02, 2.6472e-02, 2.7275e-01, 1.8995e-01,
                                        -7.7340e-02, 1.2059e-02, 3.5163e-02, 1.5442e-02, 5.1417e-02,
                                        5.0993e-01, 1.2994e-01, 2.3873e-01, -7.2816e-02, 1.5850e-01,
                                        -2.0404e-01, -2.2941e-01, 2.3660e-01, 2.0418e-01, 6.7775e-02,
                                        -3.9195e-01, 3.6655e-01, 1.6498e-01, 6.4065e-02, 4.9579e-02,
                                        2.8265e-01, -5.9919e-03, 4.0163e-02, 8.9072e-02, 1.5125e-01,
                                        9.0711e-02, -1.2608e-01, -1.0413e-01, -2.1931e-01, 5.0183e-02,
                                        -3.4841e-02, -8.1449e-02, -1.1225e-01, -4.5787e-02, -7.8871e-02,
                                        3.8858e-02, 9.2660e-02, 1.5991e-01, -6.7528e-02, -6.3166e-02,
                                        -4.7824e-03, -1.3528e-01, 1.4845e-01, 2.0460e-01, -9.3238e-02,
                                        1.4902e-03, 1.1896e-01, -3.1337e-01, 2.1637e-02, 1.4990e-01,
                                        -2.1179e-03, -8.1374e-02, -1.0241e-01, -8.0754e-02, -1.4449e-01,
                                        -1.3549e-01, -7.5588e-02, -8.0083e-02, -1.4114e-01, 2.9467e-03,
                                        3.5340e-01, -4.3351e-02, 9.6934e-02, 1.3625e-01, 1.3339e-01,
                                        -1.2059e-02, -1.4325e-01, -2.1202e-01, 3.8758e-02, 2.5965e-01,
                                        -7.8454e-02, 1.5983e-01, 1.0115e-02, 2.2192e-01, -1.4043e-01,
                                        6.7966e-02, -1.4672e-01, -1.8846e-01, 1.9488e-01, 1.2942e-01,
                                        -1.3165e-02, -1.6099e-01, -9.6146e-02, 1.3439e-01, -5.0560e-02,
                                        8.2779e-02, -2.4827e-01, -7.8047e-02, -3.1163e-01, -1.7481e-01,
                                        2.1450e-01, -7.6112e-02, -1.9967e-02, 5.7099e-02, 7.7664e-02,
                                        -7.9647e-02, 3.3941e-02, 2.9551e-02, 1.4231e-01, 2.3480e-02,
                                        1.5209e-01, -2.0011e-01, 1.1153e-01, 1.2694e-01, 8.7853e-02,
                                        2.6997e-01, 1.3525e-01, 1.9541e-01, 3.4429e-03, -9.6446e-02,
                                        7.6708e-02, -3.0698e-02, -1.8507e-01, 2.5645e-01, 2.8182e-01,
                                        -1.2282e-01, -1.1017e-01, 2.2249e-01, 2.1966e-01, 3.5795e-01,
                                        1.6279e-01, 1.7276e-01, 2.1410e-01, -3.2499e-01, 5.0327e-02,
                                        7.9813e-02, -1.5915e-01, -3.6175e-02, 1.4376e-01, 2.9565e-01,
                                        6.9097e-02, -8.0661e-01, 4.9966e-02, 6.2506e-02, 1.8852e-02,
                                        -8.6921e-02, 6.0899e-02, 2.2442e-01, -1.4272e-01, -4.0656e-04,
                                        -1.2531e-01, 1.5240e-01, -6.8841e-02, 4.2114e-01, -4.4379e-02,
                                        -3.5105e-02, 1.4931e-01, -8.3358e-02, -1.0498e-01, 1.4575e-01,
                                        -1.6491e-01, 4.7820e-02, 2.5958e-01, 1.1974e-01, 1.8271e-01,
                                        1.7439e-02, -1.5855e-01, -9.0135e-02, -2.6199e-01, -2.5709e-01,
                                        6.3203e-03, 7.5823e-02])

        self.random_text_flag = random_text
        self.random_everything_flag = random_everything
        self.fixed_text_flag = fixed_text
        self.random_image_flag = random_images

        # Weight Stacks
        self.W_query = {
            "multimodal": [self.text_reduction, self.image_reduction, self.W_query_text_half_dim,
                           self.W_query_image_half_dim],
            "image": [self.W_query_image_full_dim],
        }

        self.W_key = {
            "multimodal": [self.text_reduction, self.image_reduction, self.W_key_text_half_dim,
                           self.W_key_image_half_dim],
            "image": [self.W_key_image_full_dim]
        }

    def weight_pass(self, query_text, query_image, key_text, key_image):
        inference_functions = [
            (True, True, True, True),
            (False, True, False, True),
            (False, True, True, True)
        ]

        if None in (query_image, key_image):
            raise ValueError("Query and Key image cannot be None")

        if (query_text is not None, query_image is not None, key_text is not None,
            key_image is not None) in inference_functions:
            query = self._queries_inference(query_text, query_image)
            key = self._keys_inference(key_text, key_image)
            return query, key
        else:
            raise ValueError("Invalid input")

    def _queries_inference(self, query_text, query_image):
        if query_text is None:
            output = self.W_query_image_full_dim(query_image)
        elif query_image is None:
            raise ValueError("Query image cannot be None")
        else:
            text_reduction = self.text_reduction(query_text)
            image_reduction = self.image_reduction(query_image)
            query_text_half_dim = self.W_query_text_half_dim(text_reduction)
            query_image_half_dim = self.W_query_image_half_dim(image_reduction)
            output = torch.cat((query_text_half_dim, query_image_half_dim), dim=-1)
        return output

    def _keys_inference(self, key_text, key_image):
        if key_text is None:
            output = self.W_key_image_full_dim(key_image)
        elif key_image is None:
            raise ValueError("Key image cannot be None")
        else:
            text_reduction = self.text_reduction(key_text)
            image_reduction = self.image_reduction(key_image)
            key_text_half_dim = self.W_key_text_half_dim(text_reduction)
            key_image_half_dim = self.W_key_image_half_dim(image_reduction)
            output = torch.cat((key_text_half_dim, key_image_half_dim), dim=-1)
        return output

    def forward(self, query_text, query_image, key_text, key_image):

        query_text = query_text.to(self.device)
        query_image = query_image.to(self.device)
        key_text = key_text.to(self.device)
        key_image = key_image.to(self.device)
        query, key = self.weight_pass(query_text, query_image, key_text, key_image)

        d_k = key.size()[-1]

        key_transposed = key.transpose(1, 2)
        logits = torch.matmul(query, key_transposed) / math.sqrt(d_k)
        logits = logits.squeeze()

        if len(logits.shape) <= 2:
            softmax = F.softmax(logits, dim=0)
        else:
            softmax = F.softmax(logits, dim=1)

        return softmax, logits

    def configure_optimizers(self):
        self.my_optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        optimizer = self.my_optimizer
        return [optimizer]
