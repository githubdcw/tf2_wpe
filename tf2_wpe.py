import tensorflow as tf
#from tensorflow.contrib import signal as tf.compat.v1.signal


def _batch_wrapper(inner_function, signals, num_frames, time_axis=-1):
    """Helper function to support batching with signal lenghts respected

    Args:
        inner_function (function): A function taking the cutted signals as

        signals (tuple): Signals needed for the function. Observation must be
            in the first place. All signals must have shape (batch, ..., time)
        num_frames (array): Number of frames for each batch

    Returns:
        tf.compat.v1.Tensor: Zero padded output of the function.
    """

    max_frames = tf.compat.v1.reduce_max(num_frames)

    # If we remove the batch dimension the time axis shifts by -1 if positive
    if time_axis > 0:
        time_axis -= 1

    def _single_batch(inp):
        frames = inp[-1]
        inp = inp[0]
        with tf.compat.v1.name_scope('single_batch'):

            def _pad(x):
                padding = max_frames - \
                    tf.compat.v1.minimum(frames, tf.compat.v1.shape(x)[time_axis])
                zeros = tf.compat.v1.cast(tf.compat.v1.zeros(()), x.dtype)
                paddings = x.shape.ndims * [(0, 0), ]
                paddings[time_axis] = (0, padding)
                return tf.compat.v1.pad(
                    x,
                    paddings,
                    constant_values=zeros
                )

            def _slice(x):
                slices = x.shape.ndims * [slice(None), ]
                slices[time_axis] = slice(frames)
                return x[slices]

            enhanced = inner_function(
                [_slice(i) for i in inp]
            )
            return _pad(enhanced)

    out = tf.compat.v1.map_fn(
        _single_batch, [signals, num_frames], dtype=signals[0].dtype
    )
    out.set_shape(signals[0].shape)
    return out


def get_power_online(signal):
    """Calculates power for `signal`

        Args:
            signal (tf.compat.v1.Tensor): Signal with shape (F, D, T).

        Returns:
            tf.compat.v1.Tensor: Power with shape (F,)

    """
    power_estimate = get_power(signal)
    power_estimate = tf.compat.v1.reduce_mean(power_estimate, axis=-1)
    return power_estimate


def get_power_inverse(signal):
    """Calculates inverse power for `signal`

    Args:
        signal (tf.compat.v1.Tensor): Single frequency signal with shape (D, T).
        psd_context: context for power estimation
    Returns:
        tf.compat.v1.Tensor: Inverse power with shape (T,)

    """
    power = get_power(signal)
    eps = 1e-10 * tf.compat.v1.reduce_max(power)
    inverse_power = tf.compat.v1.reciprocal(tf.compat.v1.maximum(power, eps))
    return inverse_power


def get_power(signal, axis=-2):
    """Calculates power for `signal`

    Args:
        signal (tf.compat.v1.Tensor): Single frequency signal with shape (D, T) or (F, D, T).
        axis: reduce_mean axis
    Returns:
        tf.compat.v1.Tensor: Power with shape (T,) or (F, T)

    """
    power = tf.compat.v1.real(signal) ** 2 + tf.compat.v1.imag(signal) ** 2
    power = tf.compat.v1.reduce_mean(power, axis=axis)

    return power


#def get_power(signal, psd_context=0):
#    """
#    Calculates power for single frequency signal.
#    In case psd_context is an tuple the two values
#    are describing the left and right hand context.
#
#    Args:
#        signal: (D, T)
#        psd_context: tuple or int
#    """
#    shape = tf.compat.v1.shape(signal)
#    if len(signal.get_shape()) == 2:
#        signal = tf.compat.v1.reshape(signal, (1, shape[0], shape[1]))
#
#    power = tf.compat.v1.reduce_mean(
#        tf.compat.v1.real(signal) ** 2 + tf.compat.v1.imag(signal) ** 2,
#        axis=-2
#    )
#
#    if psd_context is not 0:
#        if isinstance(psd_context, tuple):
#            context = psd_context[0] + 1 + psd_context[1]
#        else:
#            context = 2 * psd_context + 1
#            psd_context = (psd_context, psd_context)
#
#        power = tf.compat.v1.pad(
#            power,
#            ((0, 0), (psd_context[0], psd_context[1])),
#            mode='constant'
#        )
#        print(power)
#        power = tf.compat.v1.nn.convolution(
#            power,
#            tf.compat.v1.ones(context),
#            padding='VALID'
#        )[psd_context[1]:-psd_context[0]]
#
#        denom = tf.compat.v1.nn.convolution(
#            tf.compat.v1.zeros_like(power) + 1.,
#            tf.compat.v1.ones(context),
#            padding='VALID'
#        )[psd_context[1]:-psd_context[0]]
#        print(power)
#        power /= denom
#
#    elif psd_context == 0:
#        pass
#    else:
#        raise ValueError(psd_context)
#
#    return tf.compat.v1.squeeze(power, axis=0)


def get_correlations(Y, inverse_power, taps, delay):
    """Calculates weighted correlations of a window of length taps

    Args:
        Y (tf.compat.v1.Ttensor): Complex-valued STFT signal with shape (F, D, T)
        inverse_power (tf.compat.v1.Tensor): Weighting factor with shape (F, T)
        taps (int): Lenghts of correlation window
        delay (int): Delay for the weighting factor

    Returns:
        tf.compat.v1.Tensor: Correlation matrix of shape (F, taps*D, taps*D)
        tf.compat.v1.Tensor: Correlation vector of shape (F, taps*D)
    """
    dyn_shape = tf.compat.v1.shape(Y)
    F = dyn_shape[0]
    D = dyn_shape[1]
    T = dyn_shape[2]

    Psi = tf.compat.v1.signal.frame(Y, taps, 1, axis=-1)[..., :T - delay - taps + 1, ::-1]
    Psi_conj_norm = (
        tf.compat.v1.cast(inverse_power[:, None, delay + taps - 1:, None], Psi.dtype)
        * tf.compat.v1.conj(Psi)
    )

    correlation_matrix = tf.compat.v1.einsum('fdtk,fetl->fkdle', Psi_conj_norm, Psi)
    correlation_vector = tf.compat.v1.einsum(
        'fdtk,fet->fked', Psi_conj_norm, Y[..., delay + taps - 1:]
    )

    correlation_matrix = tf.compat.v1.reshape(correlation_matrix, (F, taps * D, taps * D))
    return correlation_matrix, correlation_vector


def get_correlations_for_single_frequency(Y, inverse_power, taps, delay):
    """Calculates weighted correlations of a window of length taps for one freq.

    Args:
        Y (tf.compat.v1.Ttensor): Complex-valued STFT signal with shape (D, T)
        inverse_power (tf.compat.v1.Tensor): Weighting factor with shape (T)
        K (int): Lenghts of correlation window
        delay (int): Delay for the weighting factor

    Returns:
        tf.compat.v1.Tensor: Correlation matrix of shape (taps*D, taps*D)
        tf.compat.v1.Tensor: Correlation vector of shape (D, taps*D)
    """
    correlation_matrix, correlation_vector = get_correlations(
        Y[None], inverse_power[None], taps, delay
    )
    return correlation_matrix[0], correlation_vector[0]


def get_filter_matrix_conj(
        Y, correlation_matrix, correlation_vector, taps, delay, mode='solve'):
    """Calculate (conjugate) filter matrix based on correlations for one freq.

    Args:
        Y (tf.compat.v1.Tensor): Complex-valued STFT signal of shape (D, T)
        correlation_matrix (tf.compat.v1.Tensor): Correlation matrix (taps*D, taps*D)
        correlation_vector (tf.compat.v1.Tensor): Correlation vector (D, taps*D)
        K (int): Number of filter taps
        delay (int): Delay
        mode (str, optional): Specifies how R^-1@r is calculate:
            "inv" calculates the inverse of R directly and then uses matmul
            "solve" solves Rx=r for x

    Raises:
        ValueError: Unknown mode specified

    Returns:
        tf.compat.v1.Tensor: (Conjugate) filter Matrix
    """

    D = tf.compat.v1.shape(Y)[0]

    correlation_vector = tf.compat.v1.reshape(correlation_vector, (D * D * taps, 1))
    selector = \
        tf.compat.v1.reshape(
            tf.compat.v1.transpose(
                tf.compat.v1.reshape(tf.compat.v1.range(D * D * taps), (D, taps, D)), (1, 0, 2)), (-1,))
    inv_selector = \
        tf.compat.v1.reshape(
            tf.compat.v1.transpose(
                tf.compat.v1.reshape(tf.compat.v1.range(D * D * taps), (taps, D, D)), (1, 0, 2)), (-1,))

    correlation_vector = tf.compat.v1.gather(correlation_vector, inv_selector)

    if mode == 'inv':
        with tf.compat.v1.device('/cpu:0'):
            inv_correlation_matrix = tf.compat.v1.matrix_inverse(correlation_matrix)
        stacked_filter_conj = tf.compat.v1.einsum(
            'ab,cb->ca',
            inv_correlation_matrix, tf.compat.v1.reshape(correlation_vector, (D, D * taps))
        )
        stacked_filter_conj = tf.compat.v1.reshape(stacked_filter_conj, (D * D * taps, 1))
    elif mode == 'solve':
        with tf.compat.v1.device('/cpu:0'):
            stacked_filter_conj = tf.compat.v1.reshape(
                tf.compat.v1.matrix_solve(
                    tf.compat.v1.tile(correlation_matrix[None, ...], [D, 1, 1]),
                    tf.compat.v1.reshape(correlation_vector, (D, D * taps, 1))
                ),
                (D * D * taps, 1)
            )
    else:
        raise ValueError(
            'Unknown mode {}. Possible are "inv" and solve"'.format(mode))
    stacked_filter_conj = tf.compat.v1.gather(stacked_filter_conj, selector)

    filter_matrix_conj = tf.compat.v1.transpose(
        tf.compat.v1.reshape(stacked_filter_conj, (taps, D, D)),
        (0, 2, 1)
    )
    return filter_matrix_conj


def perform_filter_operation(Y, filter_matrix_conj, taps, delay):
    """

    >>> D, T, taps, delay = 1, 10, 2, 1
    >>> tf.compat.v1.enable_eager_execution()
    >>> Y = tf.compat.v1.ones([D, T])
    >>> filter_matrix_conj = tf.compat.v1.ones([taps, D, D])
    >>> X = perform_filter_operation_v2(Y, filter_matrix_conj, taps, delay)
    >>> X.shape
    TensorShape([Dimension(1), Dimension(10)])
    >>> X.numpy()
    array([[ 1.,  0., -1., -1., -1., -1., -1., -1., -1., -1.]], dtype=float32)
    """
    dyn_shape = tf.compat.v1.shape(Y)
    T = dyn_shape[1]

    def add_tap(accumulated, tau_minus_delay):
        new = tf.compat.v1.einsum(
            'de,dt',
            filter_matrix_conj[tau_minus_delay, :, :],
            Y[:, :(T - delay - tau_minus_delay)]
        )
        paddings = tf.compat.v1.convert_to_tensor([[0, 0], [delay + tau_minus_delay, 0]])
        new = tf.compat.v1.pad(new, paddings, "CONSTANT")
        return accumulated + new

    reverb_tail = tf.compat.v1.foldl(
        add_tap, tf.compat.v1.range(0, taps),
        initializer=tf.compat.v1.zeros_like(Y)
    )
    return Y - reverb_tail


def single_frequency_wpe(Y, taps=10, delay=3, iterations=3, mode='inv'):
    """WPE for a single frequency.

    Args:
        Y: Complex valued STFT signal with shape (D, T)
        taps: Number of filter taps
        delay: Delay as a guard interval, such that X does not become zero.
        iterations:
        mode (str, optional): Specifies how R^-1@r is calculate:
            "inv" calculates the inverse of R directly and then uses matmul
            "solve" solves Rx=r for x

    Returns:

    """

    enhanced = Y
    for _ in range(iterations):
        inverse_power = get_power_inverse(enhanced)
        correlation_matrix, correlation_vector = \
            get_correlations_for_single_frequency(Y, inverse_power, taps, delay)
        filter_matrix_conj = get_filter_matrix_conj(
            Y, correlation_matrix, correlation_vector,
            taps, delay, mode=mode
        )
        enhanced = perform_filter_operation(Y, filter_matrix_conj, taps, delay)
    return enhanced, inverse_power


def wpe(Y, taps=10, delay=3, iterations=3, mode='inv'):
    """WPE for all frequencies at once. Use this for regular processing.

    Args:
        Y (tf.compat.v1.Tensor): Observed signal with shape (F, D, T)
        num_frames (tf.compat.v1.Tensor): Number of frames for each signal in the batch
        taps (int, optional): Defaults to 10. Number of filter taps.
        delay (int, optional): Defaults to 3.
        iterations (int, optional): Defaults to 3.
        mode (str, optional): Specifies how R^-1@r is calculated:
            "inv" calculates the inverse of R directly and then uses matmul
            "solve" solves Rx=r for x

    Returns:
        tf.compat.v1.Tensor: Dereverberated signal
        tf.compat.v1.Tensor: Latest estimation of the clean speech PSD
    """

    def iteration_over_frequencies(y):
        enhanced, inverse_power = single_frequency_wpe(
            y, taps, delay, iterations, mode=mode)
        return (enhanced, inverse_power)

    enhanced, inverse_power = tf.compat.v1.map_fn(
        iteration_over_frequencies, Y, dtype=(Y.dtype, Y.dtype.real_dtype)
    )

    return enhanced


def batched_wpe(Y, num_frames, taps=10, delay=3, iterations=3, mode='inv'):
    """Batched version of iterative WPE.

    Args:
        Y (tf.compat.v1.Tensor): Observed signal with shape (B, F, D, T)
        num_frames (tf.compat.v1.Tensor): Number of frames for each signal in the batch
        taps (int, optional): Defaults to 10. Number of filter taps.
        delay (int, optional): Defaults to 3.
        iterations (int, optional): Defaults to 3.
        mode (str, optional): Specifies how R^-1@r is calculate:
            "inv" calculates the inverse of R directly and then uses matmul
            "solve" solves Rx=r for x

    Returns:
        tf.compat.v1.Tensor: Dereverberated signal of shape (B, F, D, T).
    """

    def _inner_func(signals):
        out = wpe(signals[0], taps, delay, iterations, mode)
        return out

    return _batch_wrapper(_inner_func, [Y], num_frames)


def wpe_step(Y, inverse_power, taps=10, delay=3, mode='inv', Y_stats=None):
    """Single step of 'wpe'. More suited for backpropagation.

    Args:
        Y (tf.compat.v1.Tensor): Complex valued STFT signal with shape (F, D, T)
        inverse_power (tf.compat.v1.Tensor): Power signal with shape (F, T)
        taps (int, optional): Filter order
        delay (int, optional): Delay as a guard interval, such that X does not become zero.
        mode (str, optional): Specifies how R^-1@r is calculate:
            "inv" calculates the inverse of R directly and then uses matmul
            "solve" solves Rx=r for x
        Y_stats (tf.compat.v1.Tensor or None, optional): Complex valued STFT signal
            with shape (F, D, T) use to calculate the signal statistics
            (i.e. correlation matrix/vector).
            If None, Y is used. Otherwise it's usually a segment of Y

    Returns:
        Dereverberated signal of shape (F, D, T)
    """
    with tf.compat.v1.name_scope('WPE'):
        with tf.compat.v1.name_scope('correlations'):
            if Y_stats is None:
                Y_stats = Y
            correlation_matrix, correlation_vector = get_correlations(
                Y_stats, inverse_power, taps, delay
            )

        def step(inp):
            (Y_f, correlation_matrix_f, correlation_vector_f) = inp
            with tf.compat.v1.name_scope('filter_matrix'):
                filter_matrix_conj = get_filter_matrix_conj(
                    Y_f,
                    correlation_matrix_f, correlation_vector_f,
                    taps, delay, mode=mode
                )
            with tf.compat.v1.name_scope('apply_filter'):
                enhanced = perform_filter_operation(
                    Y_f, filter_matrix_conj, taps, delay)
            return enhanced

        enhanced = tf.compat.v1.map_fn(
            step,
            (Y, correlation_matrix, correlation_vector),
            dtype=Y.dtype,
            parallel_iterations=100
        )

        return enhanced


def batched_wpe_step(
        Y, inverse_power, num_frames, taps=10, delay=3, mode='inv', Y_stats=None):
    """Batched single WPE step. More suited for backpropagation.

    Args:
        Y (tf.compat.v1.Tensor): Complex valued STFT signal with shape (B, F, D, T)
        inverse_power (tf.compat.v1.Tensor): Power signal with shape (B, F, T)
        num_frames (tf.compat.v1.Tensor): Number of frames for each signal in the batch
        taps (int, optional): Filter order
        delay (int, optional): Delay as a guard interval, such that X does not become zero.
        mode (str, optional): Specifies how R^-1@r is calculate:
            "inv" calculates the inverse of R directly and then uses matmul
            "solve" solves Rx=r for x
        Y_stats (tf.compat.v1.Tensor or None, optional): Complex valued STFT signal
            with shape (F, D, T) use to calculate the signal statistics
            (i.e. correlation matrix/vector).
            If None, Y is used. Otherwise it's usually a segment of Y

    Returns:
        Dereverberated signal of shape B, (F, D, T)
    """
    def _inner_func(signals):
        _Y, _inverse_power, _Y_stats = signals
        out = wpe_step(_Y, _inverse_power, taps, delay, mode, _Y_stats)
        return out

    if Y_stats is None:
        Y_stats = Y

    return _batch_wrapper(_inner_func, [Y, inverse_power, Y_stats], num_frames)


def block_wpe_step(
        Y, inverse_power, taps=10, delay=3, mode='inv',
        block_length_in_seconds=2., forgetting_factor=0.7,
        fft_shift=256, sampling_rate=16000):
    """Applies wpe in a block-wise fashion.

    Args:
        Y (tf.compat.v1.Tensor): Complex valued STFT signal with shape (F, D, T)
        inverse_power (tf.compat.v1.Tensor): Power signal with shape (F, T)
        taps (int, optional): Defaults to 10.
        delay (int, optional): Defaults to 3.
        mode (str, optional): Specifies how R^-1@r is calculate:
            "inv" calculates the inverse of R directly and then uses matmul
            "solve" solves Rx=r for x
        block_length_in_seconds (float, optional): Length of each block in
            seconds
        forgetting_factor (float, optional): Forgetting factor for the signal
            statistics between the blocks
        fft_shift (int, optional): Shift used for the STFT.
        sampling_rate (int, optional): Sampling rate of the observed signal.
    """
    frames_per_block = block_length_in_seconds * sampling_rate // fft_shift
    frames_per_block = tf.compat.v1.cast(frames_per_block, tf.compat.v1.int32)
    framed_Y = tf.compat.v1.signal.frame(
        Y, frames_per_block, frames_per_block, pad_end=True)
    framed_inverse_power = tf.compat.v1.signal.frame(
        inverse_power, frames_per_block, frames_per_block, pad_end=True)
    num_blocks = tf.compat.v1.shape(framed_Y)[-2]

    enhanced_arr = tf.compat.v1.TensorArray(
        framed_Y.dtype, size=num_blocks, clear_after_read=True)
    start_block = tf.compat.v1.constant(0)
    correlation_matrix, correlation_vector = get_correlations(
        framed_Y[..., start_block, :], framed_inverse_power[..., start_block, :],
        taps, delay
    )
    num_bins = Y.shape[0]
    num_channels = Y.shape[1].value
    if num_channels is None:
        num_channels = tf.compat.v1.shape(Y)[1]
    num_frames = tf.compat.v1.shape(Y)[-1]

    def cond(k, *_):
        return k < num_blocks

    with tf.compat.v1.name_scope('block_WPE'):
        def block_step(
                k, enhanced, correlation_matrix_tm1, correlation_vector_tm1):

            def _init_step():
                return correlation_matrix_tm1, correlation_vector_tm1

            def _update_step():
                correlation_matrix, correlation_vector = get_correlations(
                    framed_Y[..., k, :], framed_inverse_power[..., k, :],
                    taps, delay
                )
                return (
                    (1. - forgetting_factor) * correlation_matrix_tm1
                    + forgetting_factor * correlation_matrix,
                    (1. - forgetting_factor) * correlation_vector_tm1
                    + forgetting_factor * correlation_vector
                )

            correlation_matrix, correlation_vector = tf.compat.v1.case(
                ((tf.compat.v1.equal(k, 0), _init_step),), default=_update_step
            )

            def step(inp):
                (Y_f, inverse_power_f,
                    correlation_matrix_f, correlation_vector_f) = inp
                with tf.compat.v1.name_scope('filter_matrix'):
                    filter_matrix_conj = get_filter_matrix_conj(
                        Y_f,
                        correlation_matrix_f, correlation_vector_f,
                        taps, delay, mode=mode
                    )
                with tf.compat.v1.name_scope('apply_filter'):
                    enhanced_f = perform_filter_operation(
                        Y_f, filter_matrix_conj, taps, delay)
                return enhanced_f

            enhanced_block = tf.compat.v1.map_fn(
                step,
                (framed_Y[..., k, :], framed_inverse_power[..., k, :],
                 correlation_matrix, correlation_vector),
                dtype=framed_Y.dtype,
                parallel_iterations=100
            )

            enhanced = enhanced.write(k, enhanced_block)
            return k + 1, enhanced, correlation_matrix, correlation_vector

        _, enhanced_arr, _, _ = tf.compat.v1.while_loop(
            cond, block_step,
            (start_block, enhanced_arr, correlation_matrix, correlation_vector)
        )

        enhanced = enhanced_arr.stack()
        enhanced = tf.compat.v1.transpose(enhanced, (1, 2, 0, 3))
        enhanced = tf.compat.v1.reshape(enhanced, (num_bins, num_channels, -1))

        return enhanced[..., :num_frames]


def batched_block_wpe_step(
        Y, inverse_power, num_frames, taps=10, delay=3, mode='inv',
        block_length_in_seconds=2., forgetting_factor=0.7,
        fft_shift=256, sampling_rate=16000):
    """Batched single WPE step. More suited for backpropagation.

    Args:
        Y (tf.compat.v1.Tensor): Complex valued STFT signal with shape (B, F, D, T)
        inverse_power (tf.compat.v1.Tensor): Power signal with shape (B, F, T)
        num_frames (tf.compat.v1.Tensor): Number of frames for each signal in the batch
        taps (int, optional): Filter order
        delay (int, optional): Delay as a guard interval, such that X does not become zero.
        mode (str, optional): Specifies how R^-1@r is calculate:
            "inv" calculates the inverse of R directly and then uses matmul
            "solve" solves Rx=r for x
        block_length_in_seconds (float, optional): Length of each block in
            seconds
        forgetting_factor (float, optional): Forgetting factor for the signal
            statistics between the blocks
        fft_shift (int, optional): Shift used for the STFT.
        sampling_rate (int, optional): Sampling rate of the observed signal.

    Returns:
        Dereverberated signal of shape B, (F, D, T)
    """
    def _inner_func(signals):
        _Y, _inverse_power = signals
        out = block_wpe_step(
            _Y, _inverse_power, taps, delay,
            mode, block_length_in_seconds, forgetting_factor,
            fft_shift, sampling_rate)
        return out

    return _batch_wrapper(_inner_func, [Y, inverse_power], num_frames)


def online_wpe_step(
        input_buffer, power_estimate, inv_cov, filter_taps,
        alpha, taps, delay
    ):
    """
    One step of online dereverberation

    Args:
        input_buffer (tf.compat.v1.Tensor): Buffer of shape (taps+delay+1, F, D)
        power_estimate (tf.compat.v1.Tensor): Estimate for the current PSD
        inv_cov (tf.compat.v1.Tensor): Current estimate of R^-1
        filter_taps (tf.compat.v1.Tensor): Current estimate of filter taps (F, taps*D, taps)
        alpha (float): Smoothing factor
        taps (int): Number of filter taps
        delay (int): Delay in frames

    Returns:
        tf.compat.v1.Tensor: Dereverberated frame of shape (F, D)
        tf.compat.v1.Tensor: Updated estimate of R^-1
        tf.compat.v1.Tensor: Updated estimate of the filter taps
    """
    F = input_buffer.shape[-2]
    D = tf.compat.v1.shape(input_buffer)[-1]
    window = input_buffer[:-delay - 1][::-1]
    window = tf.compat.v1.reshape(
        tf.compat.v1.transpose(window, (1, 2, 0)), (F, taps * D)
    )
    window_conj = tf.compat.v1.conj(window)
    pred = (
        input_buffer[-1] -
        tf.compat.v1.einsum('fid,fi->fd', tf.compat.v1.conj(filter_taps), window)
    )

    nominator = tf.compat.v1.einsum('fij,fj->fi', inv_cov, window)
    denominator = tf.compat.v1.cast(alpha * power_estimate, window.dtype)
    denominator += tf.compat.v1.einsum('fi,fi->f', window_conj, nominator)
    kalman_gain = nominator / denominator[:, None]

    inv_cov_k = inv_cov - tf.compat.v1.einsum('fj,fjm,fi->fim', window_conj, inv_cov, kalman_gain)
    inv_cov_k /= alpha

    filter_taps_k = (
        filter_taps +
        tf.compat.v1.einsum('fi,fm->fim', kalman_gain, tf.compat.v1.conj(pred))
    )
    return pred, inv_cov_k, filter_taps_k


def recursive_wpe(
        Y, power_estimate, alpha, taps=10, delay=2,
        only_use_final_filters=False):
    """Applies WPE in a framewise recursive fashion.

    Args:
        Y (tf.compat.v1.Tensor): Observed signal of shape (T, F, D)
        power_estimate (tf.compat.v1.Tensor): Estimate for the clean signal PSD of shape (T, F)
        alpha (float): Smoothing factor for the recursion
        taps (int, optional): Number of filter taps.
        delay (int, optional): Delay
        only_use_final_filters (bool, optional): Applies only the final
            estimated filter coefficients to the whole signal. This is for
            debugging purposes only and makes this method a offline one.

    Returns:
        tf.compat.v1.Tensor: Enhanced signal
    """

    num_frames = tf.compat.v1.shape(Y)[0]
    num_bins = Y.shape[1]
    num_ch = tf.compat.v1.shape(Y)[-1]
    dtype = Y.dtype
    k = delay + taps

    inv_cov_tm1 = tf.compat.v1.eye(num_ch * taps, batch_shape=[num_bins], dtype=dtype)
    filter_taps_tm1 = tf.compat.v1.zeros((num_bins, num_ch * taps, num_ch), dtype=dtype)
    enhanced_arr = tf.compat.v1.TensorArray(dtype, size=num_frames, name='dereverberated')
    Y = tf.compat.v1.pad(Y, ((delay + taps, 0), (0, 0), (0, 0)))

    def dereverb_step(k_, inv_cov_tm1, filter_taps_tm1, enhanced):
        pos = k_ - delay - taps
        input_buffer = Y[pos:k_ + 1]
        pred, inv_cov_k, filter_taps_k = online_wpe_step(
            input_buffer, power_estimate[pos],
            inv_cov_tm1, filter_taps_tm1, alpha, taps, delay
        )
        enhanced_k = enhanced.write(pos, pred)
        return k_ + 1, inv_cov_k, filter_taps_k, enhanced_k

    def cond(k, *_):
        return tf.compat.v1.less(k, num_frames + delay + taps)

    _, _, final_filter_taps, enhanced = tf.compat.v1.while_loop(
        cond, dereverb_step, (k, inv_cov_tm1, filter_taps_tm1, enhanced_arr))

    # Only for testing / oracle purposes
    def dereverb_with_filters(k_, filter_taps, enhanced):
        window = Y[k_ - delay - taps:k_ - delay][::-1]
        window = tf.compat.v1.reshape(
            tf.compat.v1.transpose(window, (1, 2, 0)), (-1, taps * num_ch)
        )
        pred = (
            Y[k_] -
            tf.compat.v1.einsum('lim,li->lm', tf.compat.v1.conj(filter_taps), window)
        )
        enhanced_k = enhanced.write(k_ - delay - taps, pred)
        return k_ + 1, filter_taps, enhanced_k

    if only_use_final_filters:
        k = tf.compat.v1.constant(0) + delay + taps
        enhanced_arr = tf.compat.v1.TensorArray(dtype, size=num_frames)
        _, _, enhanced = tf.compat.v1.while_loop(
            cond, dereverb_with_filters, (k, final_filter_taps, enhanced_arr))

    return enhanced.stack()


def batched_recursive_wpe(
        Y, power_estimate, alpha, num_frames, taps=10, delay=2,
        only_use_final_filters=False):
    """Batched single WPE step. More suited for backpropagation.

    Args:
        Y (tf.compat.v1.Tensor): Observed signal of shape (B, T, F, D)
        power_estimate (tf.compat.v1.Tensor): Estimate for the clean signal PSD of shape (B, T, F)
        alpha (float): Smoothing factor for the recursion
        num_frames (tf.compat.v1.Tensor): Number of frames for each signal in the batch
        K (int, optional): Number of filter taps.
        delay (int, optional): Delay
        only_use_final_filters (bool, optional): Applies only the final
            estimated filter coefficients to the whole signal. This is for
            debugging purposes only and makes this method a offline one.

    Returns:
        Dereverberated signal of shape (B, T, F, D)
    """
    def _inner_func(signals):
        _Y, _power_estimate = signals
        out = recursive_wpe(
            _Y, _power_estimate, alpha, taps, delay, only_use_final_filters)
        return out

    return _batch_wrapper(
        _inner_func, [Y, power_estimate], num_frames, time_axis=1)
