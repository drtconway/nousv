use arrow::array::{ArrayRef, Int32Array, UInt8Array, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::collections::HashMap;
use std::sync::Arc;

use noodles::vcf;
use noodles::vcf::Header;
use noodles::vcf::variant::record::Samples;
use noodles::vcf::variant::{Record, RecordBuf};
use std::fs::File;
use std::io::{self, BufRead, Error};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariantFeatureKind {
    Insertion,
    Deletion,
    Duplication,
    Inversion,
    Translocation,
}

impl TryFrom<&str> for VariantFeatureKind {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "INS" => Ok(Self::Insertion),
            "DEL" => Ok(Self::Deletion),
            "DUP" => Ok(Self::Duplication),
            "INV" => Ok(Self::Inversion),
            "TRA" | "BND" => Ok(Self::Translocation),
            _ => Err(Error::new(
                io::ErrorKind::Other,
                "Unknown variant feature kind",
            )),
        }
    }
}

pub fn extract_variant_features(vcf_path: &str, output_path: &str) -> std::io::Result<()> {
    let file = File::open(vcf_path)?;
    let reader = io::BufReader::new(file);
    let variant_iter = VariantFeatureIter::new(reader)?;

    save_variant_features_to_parquet(variant_iter, output_path)?;

    Ok(())
}

pub fn save_variant_features_to_parquet<I>(features: I, path: &str) -> std::io::Result<()>
where
    I: IntoIterator<Item = std::io::Result<(usize, i32, i32, VariantFeatureKind, Vec<i32>)>>,
{
    let mut chrom_ids = Vec::new();
    let mut begins = Vec::new();
    let mut ends = Vec::new();
    let mut kinds = Vec::new();
    let mut refs = Vec::new();
    let mut alts = Vec::new();

    for item in features {
        let (chrom_id, begin, end, kind, alleles) = item?;
        chrom_ids.push(chrom_id as u32);
        begins.push(begin);
        ends.push(end);
        kinds.push(kind as u8);
        refs.push(*alleles.get(0).unwrap_or(&0));
        alts.push(*alleles.get(1).unwrap_or(&0));
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("chrom", DataType::UInt32, false),
        Field::new("begin", DataType::Int32, false),
        Field::new("end", DataType::Int32, false),
        Field::new("kind", DataType::UInt8, false),
        Field::new("ref", DataType::Int32, false),
        Field::new("alt", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(chrom_ids)) as ArrayRef,
            Arc::new(Int32Array::from(begins)) as ArrayRef,
            Arc::new(Int32Array::from(ends)),
            Arc::new(UInt8Array::from(kinds)),
            Arc::new(Int32Array::from(refs)),
            Arc::new(Int32Array::from(alts)),
        ],
    )
    .map_err(|error| Error::new(io::ErrorKind::Other, error))?;

    let file = File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

pub struct VariantFeatureIter<R: io::BufRead> {
    reader: vcf::io::Reader<R>,
    header: Header,
    chrom_names: Vec<String>,
    chrom_index: HashMap<String, usize>,
    record: RecordBuf,
}

impl<R: io::BufRead> VariantFeatureIter<R> {
    pub fn new(source: R) -> io::Result<Self> {
        let mut reader = vcf::io::Reader::new(source);
        let header = reader.read_header()?;
        let chrom_names: Vec<String> = header
            .contigs()
            .iter()
            .map(|(name, _)| name.to_string())
            .collect::<Vec<_>>();
        let chrom_index: HashMap<String, usize> = chrom_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        Ok(Self {
            reader,
            header,
            chrom_names,
            chrom_index,
            record: RecordBuf::default(),
        })
    }

    fn next_impl(
        &mut self,
    ) -> std::io::Result<Option<(usize, i32, i32, VariantFeatureKind, Vec<i32>)>> {
        let n = self
            .reader
            .read_record_buf(&mut self.header, &mut self.record)?;
        if n == 0 {
            return Ok(None);
        }
        let chrom = self.record.reference_sequence_name();
        let chrom_id = *self.chrom_index.get(chrom).unwrap();
        let begin = self.record.variant_start().unwrap().get() as i32;
        let end = self.record.variant_end(&self.header)?.get() as i32;
        let svtype = info_field_as_string(&self.record, &self.header, "SVTYPE")?.unwrap();
        let svtype = VariantFeatureKind::try_from(svtype.as_str())?;
        let samples = self.record.samples();
        let sample_values = match samples.iter().next() {
            Some(s) => s,
            None => return Err(io::Error::new(io::ErrorKind::Other, "No samples")),
        };
        let gt = match sample_values.get(&self.header, "GT").unwrap()?.unwrap() {
            vcf::variant::record::samples::series::Value::Genotype(gt) => gt,
            _ => return Err(io::Error::new(io::ErrorKind::Other, "No GT")),
        };
        let mut alleles: Vec<i32> = vec![0; 2];
        for allele in gt.iter() {
            match allele {
                Ok(a) => {
                    if let Some(idx) = a.0 {
                        if (idx as usize) < alleles.len() {
                            alleles[idx as usize] += 1;
                        }
                    }
                }
                Err(e) => return Err(e),
            }
        }
        Ok(Some((chrom_id, begin, end, svtype, alleles)))
    }

    #[allow(dead_code)]
    pub fn names(&self) -> &[String] {
        &self.chrom_names
    }

}

impl<R: BufRead> Iterator for VariantFeatureIter<R> {
    type Item = io::Result<(usize, i32, i32, VariantFeatureKind, Vec<i32>)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_impl().transpose()
    }
}

fn info_field_as_string<Rec: Record>(
    record: &Rec,
    header: &Header,
    field: &str,
) -> std::io::Result<Option<String>> {
    match record.info().get(header, field) {
        Some(value) => {
            let value = value?;
            match value {
                Some(value) => match value {
                    vcf::variant::record::info::field::Value::Integer(_)
                    | vcf::variant::record::info::field::Value::Float(_)
                    | vcf::variant::record::info::field::Value::Flag
                    | vcf::variant::record::info::field::Value::Character(_) => Err(Error::new(
                        io::ErrorKind::Other,
                        "Expected string info value",
                    )),
                    vcf::variant::record::info::field::Value::String(cow) => {
                        Ok(Some(cow.to_string()))
                    }
                    vcf::variant::record::info::field::Value::Array(_) => Err(Error::new(
                        io::ErrorKind::Other,
                        "Expected string info value, but got array",
                    )),
                },
                None => Ok(None),
            }
        }
        None => Ok(None),
    }
}
