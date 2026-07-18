use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

pub(crate) struct LinearW {
    inner: Linear,
    /// Optional LoRA delta: (A [rank, in_features], B [out_features, rank], scale=alpha/rank).
    /// forward output = base(x) + scale * x @ A.T @ B.T
    lora: Option<(Tensor, Tensor, f64)>,
}

impl LinearW {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { inner: Linear::new(weight, bias), lora: None }
    }

    pub fn set_lora(&mut self, a: Tensor, b: Tensor, scale: f64) {
        self.lora = Some((a, b, scale));
    }

    pub fn clear_lora(&mut self) {
        self.lora = None;
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base = self.inner.forward(x)?;
        match &self.lora {
            None => Ok(base),
            Some((a, b, scale)) => {
                // x: [..., in_f]; a: [rank, in_f]; b: [out_f, rank]
                // delta = x @ a.T @ b.T * scale  →  [..., out_f]
                // broadcast_matmul handles rank mismatch (x is 3D, a/b are 2D)
                let delta = (x.broadcast_matmul(&a.t()?)?.broadcast_matmul(&b.t()?)? * *scale)?;
                (base + delta).map_err(Into::into)
            }
        }
    }
}
