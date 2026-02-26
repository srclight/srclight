# GPU Multi-Backend Support Investigation

**Date:** 2026-02-24
**Repo:** srclight
**Goal:** Add support for AMD GPUs (including old ones) and Apple Silicon/Mac GPUs

---

## Current State

The `vector_math.py` module currently supports:
1. **CuPy (GPU)** - NVIDIA CUDA only
2. **NumPy (CPU)** - Fallback
3. **Pure Python** - Last resort

```python
# Current backend detection (lines 10-22)
try:
    import cupy as _np
    _np.zeros(1)  # Test GPU availability
    _backend = "cupy"
except Exception:
    try:
        import numpy as _np
        _backend = "numpy"
    except ImportError:
        _backend = "python"
```

---

## Target GPUs

### 1. AMD GPUs (ROCm)
- **Old AMD cards:** RX 580, RX 5700, etc. (GCN 3.0+)
- **New AMD cards:** RX 7900 XTX, RX 7600, etc. (RDNA 2/3)
- **CDNA:** Instinct MI series (for compute)

**CuPy Status:** ✅ **Experimental support exists!**
- Package: `cupy-rocm-7-0` (also 6.x available)
- Requires ROCm 7.0+ driver stack
- Install: `pip install cupy-rocm-7-0`

**Challenge:** Old cards (RX 580, RX 480, etc.) may not work with ROCm 7.0 - they need older ROCm versions (5.x/6.x) which have limited CuPy support.

### 2. Apple Silicon / Mac GPUs (MPS)
- **M1/M2/M3** Macs (all variants)
- **Intel Macs** with discrete GPUs (rare but exists)

**CuPy Status:** ❌ **No official support**
- No `cupy-metal` package exists

**Alternative:** PyTorch MPS backend
- `torch.backends.mps.is_available()` - check availability
- `device="mps"` - move tensors to Apple GPU
- Same NumPy-compatible API as CUDA

### 3. Intel GPUs (oneAPI/OpenVINO)
- **Arc** discrete GPUs
- **Integrated** (Iris Xe, etc.)
- **Data Center:** Ponte Vecchio

**CuPy Status:** ❌ No official support

**Alternative:** Intel oneAPI NumPy/DPCT or OpenVINO

---

## Implementation Options

### Option A: CuPy ROCm + PyTorch MPS (Recommended)

**Approach:**
1. Update `vector_math.py` to try more backends in order:
   - CUDA CuPy → ROCm CuPy → PyTorch MPS → NumPy → Python
   
2. Add optional dependency: `torch` (for MPS)

3. Modify backend detection to:
   ```python
   # Try CUDA CuPy first
   try:
       import cupy as _np
       _np.cuda.Device()  # Test CUDA
       _backend = "cupy_cuda"
   except Exception:
       pass
   
   # Try ROCm CuPy
   try:
       import cupyx.scipy as cupyx_sp
       import cupy as _np
       # Test ROCm via HIP
       _backend = "cupy_rocm"
   except Exception:
       pass
   
   # Try PyTorch MPS
   try:
       import torch
       if torch.backends.mps.is_available():
           _backend = "mps"
   except Exception:
       pass
   
   # Fallback to NumPy
   ```

**Challenges:**
- Different API for some operations (MPS doesn't support everything CUDA does)
- CuPy ROCm is experimental - may have bugs
- Old AMD cards need older ROCm versions

### Option B: Abstract Backend with Pluggable Strategy

**Approach:**
1. Create `backends/` module:
   - `backends/cuda.py` - NVIDIA CuPy
   - `backends/rocm.py` - AMD CuPy
   - `backends/mps.py` - Apple MPS via PyTorch
   - `backends/cpu.py` - NumPy

2. Runtime detection picks best available

**Pros:** Cleaner architecture, easier to test each backend
**Cons:** More initial code

### Option C: Keep It Simple - Just ROCm Support First

**Approach:**
1. Change CuPy import to try both CUDA and ROCm packages
2. The same `cupy` package works for both after install
3. Just need correct `cupy-rocm-XX` package installed

**Pros:** Minimal code change
**Cons:** Still no Apple GPU support

---

## Changes Required

### Files to Modify

1. **`vector_math.py`**
   - Expand backend detection
   - Handle potential API differences
   - Add MPS-specific path for some operations

2. **`vector_cache.py`**
   - Already handles backend abstraction via `_backend` check
   - May need minor tweaks for MPS

3. **`pyproject.toml`**
   - Add optional deps: `torch` (for MPS)
   - Document ROCm install: `cupy-rocm-7-0`

4. **Documentation**
   - Install guide for AMD: `pip install cupy-rocm-7-0`
   - Install guide for Mac: `pip install torch` (MPS built-in)

---

## Testing Strategy

1. **AMD GPU (ROCm):** 
   - Need machine with AMD GPU + ROCm driver
   - Or use cloud GPU (RunPod, Lambda Labs - AMD instances)

2. **Apple MPS:**
   - Any M1/M2/M3 Mac
   - `torch.backends.mps.is_available()` to verify

3. **Old AMD Cards:**
   - RX 580/480: ROCm 5.x needed, may not have CuPy wheels
   - Test and document limitations

---

## Risk Assessment

| GPU Type | Feasibility | Effort | Stability |
|----------|-------------|--------|-----------|
| AMD (new) | High | Medium | Good (CuPy exp.) |
| AMD (old) | Low | High | Poor (ROCm 5.x) |
| Apple MPS | High | Medium | Good |
| Intel Arc | Low | High | Experimental |

---

## Recommendation

**Proceed with Option A (CuPy ROCm + PyTorch MPS):**

1. First add AMD ROCm support (most straightforward)
2. Then add Apple MPS as secondary target
3. Document limitations for old AMD cards

**Estimated effort:** 2-4 hours for core implementation
**Testing:** Need physical hardware (AMD Mac or AMD GPU machine)

---

## Open Questions for Discussion

1. Which GPUs do you want to prioritize?
2. Do you have access to AMD GPU or Apple Silicon for testing?
3. Should we support Intel Arc GPUs?
4. How important is backward compatibility with old AMD cards (RX 580 era)?
