use std::error::Error;

use clap::{Parser, Subcommand};

mod event;
mod extract;
mod train;
mod utils;

#[derive(Parser)]
#[command(name = "nousv")]
#[command(about = "A tool for BAM file feature extraction", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Extract features from a BAM file
    ExtractFeatures {
        /// Input BAM file
        #[arg(short, long)]
        bam: String,

        /// Output features file (Parquet format)
        #[arg(short, long)]
        output: String,
    },

    /// Extract features from a VCF file
    ExtractVcfFeatures {
        /// Input VCF file
        #[arg(short, long)]
        vcf: String,

        /// Output Parquet file
        #[arg(short, long)]
        output: String,
    },

    /// Train the autoencoder on the features
    TrainAutoencoder {
        /// Input features file (Parquet format)
        #[arg(short, long)]
        input: String,
    },
}

fn main() -> std::io::Result<()> {
    env_logger::builder()
    .filter_level(log::LevelFilter::Info)
    .init();

    let cli = Cli::parse();

    let res: std::io::Result<()> = match &cli.command {
        Commands::ExtractFeatures { bam, output } => {
            extract::extract_features(bam, output)
        }
        Commands::ExtractVcfFeatures { vcf, output } => {
            extract::extract_variant_features(vcf, output)
        }
        Commands::TrainAutoencoder { input } => {
            train::auto::train_vae_from_parquet(input)
        }
    };
    match res {
        Ok(_) => {}
        Err(error) => {
            log::error!("Error: {}", error);
            let mut current_source = error.source();
            while let Some(source) = current_source {
                log::error!("Caused by: {}", source);
                current_source = source.source();
            }
            std::process::exit(1);
        }
    }
    Ok(())
}
