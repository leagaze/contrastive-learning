def contrastive_loss(xi, xj,  tau=1, normalize=False):
        ''' this loss is the modified torch implementation by M Diephuis here: https://github.com/mdiephuis/SimCLR/
        the inputs:
        xi, xj: image features extracted from a batch of images 2N, composed of N matching paints
        tau: temperature parameter
        normalize: normalize or not. seem to not be very useful, so better to try without.
        '''

        x = tf.keras.backend.concatenate((xi, xj), axis=0)

        sim_mat = tf.keras.backend.dot(x, tf.keras.backend.transpose(x))

        if normalize:
            sim_mat_denom = tf.keras.backend.dot(tf.keras.backend.l2_normalize(x, axis=1).unsqueeze(1), tf.keras.backend.l2_normalize(x, axis=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = tf.keras.backend.exp(sim_mat /tau)

        if normalize:
            sim_mat_denom = tf.keras.backend.l2_normalize(xi, dim=1) * tf.keras.backend.l2_normalize(xj, axis=1)
            sim_match = tf.keras.backend.exp(tf.keras.backend.sum(xi * xj, axis=-1) / sim_mat_denom / tau)
        else:
            sim_match = tf.keras.backend.exp(tf.keras.backend.sum(xi * xj, axis=-1) / tau)

        sim_match = tf.keras.backend.concatenate((sim_match, sim_match), axis=0)

        norm_sum = tf.keras.backend.exp(tf.keras.backend.ones(tf.keras.backend.shape(x)[0]) / tau)

        return tf.math.reduce_mean(-tf.keras.backend.log(sim_match / (tf.keras.backend.sum(sim_mat, axis=-1) - norm_sum)), name='contrastive_loss')
