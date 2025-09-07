use candle_core::DType;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{
    Activation, AdamW, Module, Optimizer, ParamsAdamW, Sequential, VarBuilder, VarMap, seq,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tqdm::Iter;
use std::fs::File;

struct VariationalAutoEncoder {
    encoder: Sequential,
    decoder: Sequential,
    optimizer: AdamW,
    device: Device,
}

impl VariationalAutoEncoder {
    fn new(num_features: usize, latent_dim: usize, device: &Device) -> CandleResult<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let encoder = seq()
            .add(candle_nn::linear(num_features, 128, vb.pp("ln1"))?)
            .add(Activation::Relu)
            .add(candle_nn::linear(128, latent_dim * 2, vb.pp("ln2"))?);
        let decoder = seq()
            .add(candle_nn::linear(latent_dim, 128, vb.pp("ln3"))?)
            .add(Activation::Relu)
            .add(candle_nn::linear(128, num_features, vb.pp("ln4"))?);

        let params = ParamsAdamW {
            lr: 0.0001,
            ..Default::default()
        };

        let optimizer = AdamW::new(varmap.all_vars(), params)?;

        Ok(Self {
            encoder,
            decoder,
            optimizer,
            device: device.clone(),
        })
    }

    #[allow(dead_code)]
    fn device(&self) -> &Device {
        &self.device
    }
}

pub fn train_vae_from_parquet(
    path: &str,
) -> std::io::Result<()> {
    let epochs = 10;
    let batch_size = 32;
    let latent_dim = 48;
    let device = Device::Cpu;

    // Read Parquet file into Arrow RecordBatch
    let file = File::open(path).unwrap();
    let record_batch_reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    // Collect all feature columns into a Vec<Vec<f32>>
    let mut features = Vec::new();
    for batch in record_batch_reader {
        let batch = batch.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        for row in 0..batch.num_rows() {
            let mut row_vec = Vec::new();
            for col in 2..batch.num_columns() {
                // skip chrom_id, position
                let array = batch
                    .column(col)
                    .as_any()
                    .downcast_ref::<arrow::array::Float32Array>()
                    .unwrap();
                row_vec.push(array.value(row));
            }
            features.push(row_vec);
        }
    }


    // Column-wise normalization
    let num_features = features[0].len();
    let num_samples = features.len();
    let mut means = vec![0.0f32; num_features];
    let mut stds = vec![0.0f32; num_features];

    // Compute means
    for row in &features {
        for (i, &val) in row.iter().enumerate() {
            means[i] += val;
        }
    }
    for m in &mut means {
        *m /= num_samples as f32;
    }

    // Compute stds
    for row in &features {
        for (i, &val) in row.iter().enumerate() {
            stds[i] += (val - means[i]).powi(2);
        }
    }
    for s in &mut stds {
        *s = (*s / num_samples as f32).sqrt();
        if *s == 0.0 { *s = 1.0; } // Prevent division by zero
    }

    // Normalize
    for row in &mut features {
        for (i, val) in row.iter_mut().enumerate() {
            *val = (*val - means[i]) / stds[i];
        }
    }

    let flat: Vec<f32> = features.into_iter().flatten().collect();
    let xs = Tensor::from_vec(flat, (num_samples, num_features), &device)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let mut vae = VariationalAutoEncoder::new(num_features, latent_dim, &device)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let num_batches = (num_samples + batch_size - 1) / batch_size;

    for epoch in (0..epochs).tqdm(){
        (|| {
            let mut total_loss = 0.0;
            for batch_idx in (0..num_batches).tqdm() {
                let start = batch_idx * batch_size;
                let end = ((batch_idx + 1) * batch_size).min(num_samples);
                let batch = xs.narrow(0, start, end - start)?;

                // Forward pass: encode
                let h = vae.encoder.forward(&batch)?;
                let mu = h.narrow(1, 0, latent_dim)?;
                let logvar = h.narrow(1, latent_dim, latent_dim)?;

                // Reparameterization trick
                let std = (logvar.clone() * 0.5)?.exp()?;
                let eps = Tensor::randn(0f32, 1f32, mu.shape().dims().to_vec(), &device)?;
                let z = mu.add(&std.mul(&eps)?)?;

                // Decode
                let recon = vae.decoder.forward(&z)?;

                // Loss: reconstruction + KL
                let recon_loss = recon.sub(&batch)?.powf(2.0)?.mean_all()?;
                //eprintln!("loss: {:?}", recon_loss);
                let kl = ((logvar.clone().exp()?.add(&mu.powf(2.0)?)?.neg()? + 1.0)?
                    .add(&logvar.clone())? - 0.5)?
                    .mean_all()?;
                let loss = recon_loss.add(&kl)?;

                vae.optimizer.backward_step(&loss)?;
                total_loss += loss.to_vec0::<f32>()?; // Extract scalar value from tensor
            }
            println!("epoch {epoch} loss={total_loss}");

            CandleResult::Ok(())
        })()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    }

    Ok(())
}
