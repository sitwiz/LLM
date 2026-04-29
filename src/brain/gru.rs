use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::path::Path;
use anyhow::Result;
use std::io::{Write, Read, BufWriter, BufReader};
use std::fs::File;

pub const EMBED_DIM: usize = 32;
pub const HIDDEN_DIM: usize = 64;
pub const VOCAB_SIZE: usize = 128_256;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_vec(v: &DVector<f64>) -> DVector<f64> {
    v.map(sigmoid)
}

fn tanh_vec(v: &DVector<f64>) -> DVector<f64> {
    v.map(|x| x.tanh())
}

fn softmax(v: &DVector<f64>) -> DVector<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: DVector<f64> = v.map(|x| (x - max).exp());
    let sum = exp.sum().max(1e-10);
    exp / sum
}

fn rand_matrix(rng: &mut impl Rng, rows: usize, cols: usize, scale: f64) -> DMatrix<f64> {
    DMatrix::from_fn(rows, cols, |_, _| rng.gen::<f64>() * scale - scale / 2.0)
}

fn rand_vector(rng: &mut impl Rng, size: usize, scale: f64) -> DVector<f64> {
    DVector::from_fn(size, |_, _| rng.gen::<f64>() * scale - scale / 2.0)
}

/// Save a matrix as raw binary: rows(u64) cols(u64) data(f64...)
fn write_matrix(w: &mut impl Write, m: &DMatrix<f64>) -> Result<()> {
    w.write_all(&(m.nrows() as u64).to_le_bytes())?;
    w.write_all(&(m.ncols() as u64).to_le_bytes())?;
    for v in m.iter() {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_matrix(r: &mut impl Read) -> Result<DMatrix<f64>> {
    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8)?;
    let rows = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf8)?;
    let cols = u64::from_le_bytes(buf8) as usize;
    let mut data = vec![0.0f64; rows * cols];
    for v in data.iter_mut() {
        r.read_exact(&mut buf8)?;
        *v = f64::from_le_bytes(buf8);
    }
    Ok(DMatrix::from_vec(rows, cols, data))
}

fn write_vector(w: &mut impl Write, v: &DVector<f64>) -> Result<()> {
    w.write_all(&(v.len() as u64).to_le_bytes())?;
    for x in v.iter() {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn read_vector(r: &mut impl Read) -> Result<DVector<f64>> {
    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8)?;
    let len = u64::from_le_bytes(buf8) as usize;
    let mut data = vec![0.0f64; len];
    for v in data.iter_mut() {
        r.read_exact(&mut buf8)?;
        *v = f64::from_le_bytes(buf8);
    }
    Ok(DVector::from_vec(data))
}

pub struct GRUCell {
    pub wz: DMatrix<f64>,
    pub uz: DMatrix<f64>,
    pub bz: DVector<f64>,

    pub wr: DMatrix<f64>,
    pub ur: DMatrix<f64>,
    pub br: DVector<f64>,

    pub wh: DMatrix<f64>,
    pub uh: DMatrix<f64>,
    pub bh: DVector<f64>,

    pub wy: DMatrix<f64>,
    pub by: DVector<f64>,

    pub embeddings: DMatrix<f64>,
}

impl GRUCell {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let scale = 0.1;
        Self {
            wz: rand_matrix(&mut rng, HIDDEN_DIM, EMBED_DIM, scale),
            uz: rand_matrix(&mut rng, HIDDEN_DIM, HIDDEN_DIM, scale),
            bz: rand_vector(&mut rng, HIDDEN_DIM, scale),

            wr: rand_matrix(&mut rng, HIDDEN_DIM, EMBED_DIM, scale),
            ur: rand_matrix(&mut rng, HIDDEN_DIM, HIDDEN_DIM, scale),
            br: rand_vector(&mut rng, HIDDEN_DIM, scale),

            wh: rand_matrix(&mut rng, HIDDEN_DIM, EMBED_DIM, scale),
            uh: rand_matrix(&mut rng, HIDDEN_DIM, HIDDEN_DIM, scale),
            bh: rand_vector(&mut rng, HIDDEN_DIM, scale),

            wy: rand_matrix(&mut rng, VOCAB_SIZE, HIDDEN_DIM, scale),
            by: rand_vector(&mut rng, VOCAB_SIZE, scale),

            embeddings: rand_matrix(&mut rng, VOCAB_SIZE, EMBED_DIM, scale),
        }
    }

    pub fn embed(&self, token_id: usize) -> DVector<f64> {
        self.embeddings.row(token_id.min(VOCAB_SIZE - 1)).transpose()
    }

    pub fn forward(
        &self,
        x: &DVector<f64>,
        h: &DVector<f64>,
    ) -> (DVector<f64>, DVector<f64>, DVector<f64>) {
        let z = sigmoid_vec(&(&self.wz * x + &self.uz * h + &self.bz));
        let r = sigmoid_vec(&(&self.wr * x + &self.ur * h + &self.br));
        let r_h = r.component_mul(h);
        let h_tilde = tanh_vec(&(&self.wh * x + &self.uh * r_h + &self.bh));
        let ones = DVector::from_element(HIDDEN_DIM, 1.0);
        let new_h = (&ones - &z).component_mul(h) + z.component_mul(&h_tilde);
        let logits = &self.wy * &new_h + &self.by;
        let probs = softmax(&logits);
        (new_h, logits, probs)
    }

    pub fn forward_sequence(
        &self,
        token_ids: &[usize],
    ) -> (DVector<f64>, Vec<DVector<f64>>) {
        let mut h = Self::zero_hidden();
        let mut all_probs = Vec::new();
        for &id in token_ids {
            let x = self.embed(id);
            let (new_h, _, probs) = self.forward(&x, &h);
            h = new_h;
            all_probs.push(probs);
        }
        (h, all_probs)
    }

    pub fn learn(
        &mut self,
        x: &DVector<f64>,
        h_prev: &DVector<f64>,
        target_id: usize,
        predicted_probs: &DVector<f64>,
        lr: f64,
    ) -> f64 {
        let loss = -predicted_probs[target_id].max(1e-10).ln();
        let mut d_logits = predicted_probs.clone();
        d_logits[target_id] -= 1.0;
        let dwy = &d_logits * h_prev.transpose();
        self.wy -= lr * &dwy;
        self.by -= lr * &d_logits;
        let d_h = self.wy.transpose() * &d_logits;
        let dwh = &d_h * x.transpose();
        let duh = &d_h * h_prev.transpose();
        self.wh -= lr * &dwh;
        self.uh -= lr * &duh;
        self.bh -= lr * &d_h;
        loss
    }

    pub fn update_embedding(&mut self, token_id: usize, gradient: &DVector<f64>, lr: f64) {
        let id = token_id.min(VOCAB_SIZE - 1);
        for (j, grad) in gradient.iter().enumerate() {
            self.embeddings[(id, j)] -= lr * grad;
        }
    }

    pub fn zero_hidden() -> DVector<f64> {
        DVector::zeros(HIDDEN_DIM)
    }

    /// Save weights as compact binary format
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);
        write_matrix(&mut w, &self.wz)?;
        write_matrix(&mut w, &self.uz)?;
        write_vector(&mut w, &self.bz)?;
        write_matrix(&mut w, &self.wr)?;
        write_matrix(&mut w, &self.ur)?;
        write_vector(&mut w, &self.br)?;
        write_matrix(&mut w, &self.wh)?;
        write_matrix(&mut w, &self.uh)?;
        write_vector(&mut w, &self.bh)?;
        write_matrix(&mut w, &self.wy)?;
        write_vector(&mut w, &self.by)?;
        write_matrix(&mut w, &self.embeddings)?;
        w.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);
        Ok(Self {
            wz: read_matrix(&mut r)?,
            uz: read_matrix(&mut r)?,
            bz: read_vector(&mut r)?,
            wr: read_matrix(&mut r)?,
            ur: read_matrix(&mut r)?,
            br: read_vector(&mut r)?,
            wh: read_matrix(&mut r)?,
            uh: read_matrix(&mut r)?,
            bh: read_vector(&mut r)?,
            wy: read_matrix(&mut r)?,
            by: read_vector(&mut r)?,
            embeddings: read_matrix(&mut r)?,
        })
    }

    pub fn load_or_init(path: &Path) -> Self {
        Self::load(path).unwrap_or_else(|_| {
            println!("  Initialising fresh GRU: {:?}", path);
            Self::new()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_forward_shapes() {
        let gru = GRUCell::new();
        let x = DVector::zeros(EMBED_DIM);
        let h = DVector::zeros(HIDDEN_DIM);
        let (new_h, logits, probs) = gru.forward(&x, &h);
        assert_eq!(new_h.len(), HIDDEN_DIM);
        assert_eq!(logits.len(), VOCAB_SIZE);
        assert_eq!(probs.len(), VOCAB_SIZE);
    }

    #[test]
    fn test_probs_sum_to_one() {
        let gru = GRUCell::new();
        let x = DVector::from_fn(EMBED_DIM, |i, _| i as f64 * 0.01);
        let h = DVector::zeros(HIDDEN_DIM);
        let (_, _, probs) = gru.forward(&x, &h);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hidden_state_updates() {
        let gru = GRUCell::new();
        let x = DVector::from_fn(EMBED_DIM, |i, _| i as f64 * 0.01);
        let h = DVector::zeros(HIDDEN_DIM);
        let (new_h, _, _) = gru.forward(&x, &h);
        assert!(new_h.norm() > 0.0);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let gru = GRUCell::new();
        let tmp = NamedTempFile::new().unwrap();
        gru.save(tmp.path()).unwrap();
        let loaded = GRUCell::load(tmp.path()).unwrap();
        assert_eq!(loaded.wz.nrows(), HIDDEN_DIM);
        assert_eq!(loaded.wz.ncols(), EMBED_DIM);
        assert_eq!(loaded.embeddings.nrows(), VOCAB_SIZE);
    }
}
