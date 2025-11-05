def to_hwc_uint8(frame) -> "np.ndarray":
    import numpy as np
    import torch

    # -> numpy
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)

    # remove singleton dims (e.g., (1,H,W,3), (H,W,1))
    arr = np.squeeze(arr)

    # --- Collapse to 3D and locate the channel axis robustly ---
    def _ensure_3d(a):
        import numpy as _np
        # Reduce >3D by slicing front dims until <=3D
        while a.ndim > 3:
            a = a[0]
        if a.ndim == 2:  # grayscale -> HWC
            a = _np.repeat(a[..., None], 3, axis=2)
        return a

    # If 4D, try to move the channel axis (size 1..4) to the end, then slice others
    if arr.ndim == 4:
        # find a channel-like axis (size <=4)
        ch_axes = [ax for ax, s in enumerate(arr.shape) if s in (1, 2, 3, 4)]
        if ch_axes:
            arr = np.moveaxis(arr, ch_axes[0], -1)
        # drop extra dims to get HWC
        while arr.ndim > 3:
            arr = arr[0]
    else:
        arr = _ensure_3d(arr)

    if arr.ndim != 3:
        # ultimate fallback: tiny black frame
        return np.zeros((8, 8, 3), dtype=np.uint8)

    H, W, C = arr.shape

    # If we accidentally have CHW, swap to HWC
    if C > 4 and H in (1, 2, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
        H, W, C = arr.shape

    # If still not a valid #channels, try to identify a channel axis in (H,W,C)
    if C not in (1, 2, 3, 4):
        # check if another axis is the true channel
        if H in (1, 2, 3, 4):
            arr = np.moveaxis(arr, 0, -1)  # H->C
        elif W in (1, 2, 3, 4):
            arr = np.moveaxis(arr, 1, -1)  # W->C
        H, W, C = arr.shape

    # Normalize channels
    if C == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif C == 2:
        # treat as gray+alpha-ish → use first channel and replicate to RGB
        arr = np.repeat(arr[..., :1], 3, axis=2)
    elif C > 4:
        arr = arr[..., :3]  # just take RGB

    # Scale floats if they look like [0,1]
    if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.01:
        arr = arr * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    return arr
