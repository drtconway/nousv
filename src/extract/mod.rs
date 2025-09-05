use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufReader};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, PrimitiveBuilder};
use arrow::datatypes::{DataType, Field, Float32Type, Int32Type, Schema, UInt32Type};
use arrow::record_batch::RecordBatch;
use noodles::bam;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tqdm::Iter;

use crate::event::{Event, EventKind};
pub use crate::extract::variants::extract_variant_features;
use crate::utils::heap::{Heap, HeapItem};
use bstr::BStr;

pub(crate) mod cigar;
pub(crate) mod variants;

pub fn extract_features(bam_path: &str, features_path: &str) -> io::Result<()> {
    let file = File::open(bam_path)?;
    let mut reader = bam::io::Reader::new(BufReader::new(file));

    let _header = reader.read_header()?;

    let mut writer = FeatureWriter::new(features_path);

    let mut current_contig_id = 0;
    let mut last_position = 0;
    let mut current_events = Vec::new();


    for result in reader.records().tqdm() {
        let record = result?;
        let x = hash_string(record.name().unwrap());
        if x & 0xFF != 0 {
            continue;
        }
        let flags = record.flags();
        if flags.is_unmapped() {
            continue;
        }
        let contig_id = record.reference_sequence_id().unwrap()?;
        if contig_id != current_contig_id || current_events.len() > 10000 {
            process_events(current_contig_id, current_events, &mut writer)?;
            current_contig_id = contig_id;
            last_position = 0;
            current_events = Vec::new();
        }
        let pos = record.alignment_start().unwrap()?.get() as i32;
        last_position = pos;
        let cigar = record.cigar();
        let itr = cigar::AugmentedCigarIter::new(Box::new(cigar.iter()), pos)
            .flat_map(|cig| lift(cig.map(Event::events)));
        let mut events = Vec::new();
        for event in itr {
            let event = event?;
            events.push(event);
        }
        current_events.push(events);
    }
    process_events(current_contig_id, current_events, &mut writer)?;

    drop(writer);

    Ok(())
}

fn hash_string(s: &BStr) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

fn lift<R, E>(item: Result<Vec<R>, E>) -> Vec<Result<R, E>> {
    match item {
        Ok(vec) => vec.into_iter().map(Ok).collect(),
        Err(err) => vec![Err(err)],
    }
}

fn process_events(
    contig_id: usize,
    all_events: Vec<Vec<Event>>,
    writer: &mut FeatureWriter,
) -> io::Result<()> {
    let w = 10;
    let events: Vec<Event> = CollatedEventIter::new(all_events).collect();
    let n = events.len();
    for i in 0..n {
        let left = if i >= w { i - w } else { 0 };
        let right = if i + w <= n { i + w } else { n };
        let pos = events[i].position;
        let features = process_window(w, i - left, &events[left..right]);
        writer.add_row(contig_id, pos, features);
    }
    writer.flush_current()?;
    Ok(())
}

fn process_window(width: usize, focus: usize, window: &[Event]) -> Vec<f32> {
    let pos = window[focus].position;
    let mut feature_vectors = Vec::new();
    for _i in focus..width {
        feature_vectors.push(Event::default().to_feature_vector(0));
    }
    for event in window {
        feature_vectors.push(event.to_feature_vector(pos));
    }
    while feature_vectors.len() < 2 * width {
        feature_vectors.push(Event::default().to_feature_vector(0));
    }
    let flat = feature_vectors.concat();
    flat
}

struct EventIter {
    current: Option<Event>,
    events: std::vec::IntoIter<Event>,
}

impl EventIter {
    fn new(events: Vec<Event>) -> Self {
        let mut events = events.into_iter();
        let first = events.next();
        Self {
            current: first,
            events,
        }
    }
}
impl Iterator for EventIter {
    type Item = Event;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.current.take();
        self.current = self.events.next();
        result
    }
}

impl HeapItem for EventIter {
    type KeyType = (i32, usize, EventKind); // (position, kind as usize)

    fn key(&self) -> Self::KeyType {
        let event: &Event = self.current.as_ref().unwrap();
        (event.position, event.len, event.kind)
    }
}

struct CollatedEventIter {
    heap: Heap<EventIter>,
}

impl CollatedEventIter {
    fn new(all_events: Vec<Vec<Event>>) -> Self {
        let mut heap = Heap::new();
        for events in all_events {
            let iter = EventIter::new(events);
            if iter.current.is_some() {
                heap.push(iter);
            }
        }
        Self { heap }
    }
}

impl Iterator for CollatedEventIter {
    type Item = Event;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(iter) = self.heap.pop() {
            let key = iter.key();
            let mut same = vec![iter];
            while let Some(iter1) = self.heap.front() {
                if iter1.key() != key {
                    break;
                }
                let iter1 = self.heap.pop().unwrap();
                same.push(iter1);
            }
            let events: Vec<Event> = same.iter_mut().map(|it| it.next().unwrap()).collect();
            for iter in same {
                if iter.current.is_some() {
                    self.heap.push(iter);
                }
            }
            let total_support = events.iter().map(|e| e.support).sum();
            let event = Event::new(key.2, key.0, key.1, total_support);
            return Some(event);
        }
        None
    }
}

struct FeatureWriter {
    path: String,
    inner: Option<InnerFeatureWriter>,
    current_chrom_id: usize,
    chrom_id_builder: PrimitiveBuilder<UInt32Type>,
    position_builder: PrimitiveBuilder<Int32Type>,
    feature_builder: Vec<PrimitiveBuilder<Float32Type>>,
}

impl FeatureWriter {
    pub fn new(path: &str) -> Self {
        Self {
            path: String::from(path),
            inner: None,
            current_chrom_id: 0,
            chrom_id_builder: PrimitiveBuilder::new(),
            position_builder: PrimitiveBuilder::new(),
            feature_builder: Vec::new(),
        }
    }

    pub fn add_row(&mut self, chrom_id: usize, position: i32, features: Vec<f32>) {
        if chrom_id != self.current_chrom_id {
            self.flush_current();
            self.chrom_id_builder = PrimitiveBuilder::new();
            self.position_builder = PrimitiveBuilder::new();
            self.feature_builder = Vec::new();
            self.current_chrom_id = chrom_id;
        }
        self.chrom_id_builder.append_value(chrom_id as u32);
        self.position_builder.append_value(position);
        for (i, feature) in features.into_iter().enumerate() {
            if i >= self.feature_builder.len() {
                self.feature_builder.push(PrimitiveBuilder::new());
            }
            self.feature_builder[i].append_value(feature);
        }
    }

    fn flush_current(&mut self) -> std::io::Result<()> {
        let n = self.feature_builder.len();
        let chrom_id_array = self.chrom_id_builder.finish();
        let position_array = self.position_builder.finish();
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(chrom_id_array) as ArrayRef,
            Arc::new(position_array) as ArrayRef,
        ]
        .into_iter()
        .chain(
            self.feature_builder
                .iter_mut()
                .map(|builder| Arc::new(builder.finish()) as ArrayRef),
        )
        .collect();

        let schema = self.get_schema(n)?;
        let batch = RecordBatch::try_new(schema.clone(), arrays)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;

        self.get_writer(n)?.write(&batch)?;
        self.get_writer(n)?.flush()?;

        Ok(())
    }

    fn get_schema(&mut self, n: usize) -> std::io::Result<Arc<Schema>> {
        self.make_inner(n)?;
        Ok(self.inner.as_ref().unwrap().schema.clone())
    }

    fn get_writer(&mut self, n: usize) -> std::io::Result<&mut ArrowWriter<File>> {
        self.make_inner(n)?;
        Ok(&mut self.inner.as_mut().unwrap().writer)
    }

    fn make_inner(&mut self, n: usize) -> std::io::Result<()> {
        if self.inner.is_some() {
            return Ok(());
        }
        let schema = self.make_schema(n)?;
        self.inner = Some(InnerFeatureWriter::new(&self.path, schema)?);
        Ok(())
    }

    fn make_schema(&self, n: usize) -> std::io::Result<Arc<Schema>> {
        let fields: Vec<Field> = vec![
            Field::new("chrom_id", DataType::UInt32, false),
            Field::new("position", DataType::Int32, false),
        ]
        .into_iter()
        .chain((0..n).map(|i| Field::new(&format!("feature_{}", i), DataType::Float32, false)))
        .collect();
        Ok(Arc::new(Schema::new(fields)))
    }
}

impl Drop for FeatureWriter {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            let _ = inner.writer.close();
        }
    }
}

struct InnerFeatureWriter {
    schema: Arc<Schema>,
    writer: ArrowWriter<File>,
}

impl InnerFeatureWriter {
    pub fn new(path: &str, schema: Arc<Schema>) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        Ok(Self { writer, schema })
    }

    pub fn write(&mut self, batch: &RecordBatch) -> std::io::Result<()> {
        self.writer.write(batch)?;
        Ok(())
    }
}

pub fn save_feature_vectors_to_parquet(
    feature_vectors: &[Vec<f32>],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Flatten feature vectors into columns
    if feature_vectors.is_empty() {
        return Ok(());
    }
    let num_features = feature_vectors[0].len();
    let num_rows = feature_vectors.len();

    // Transpose: columns[i] = Vec<f32> for feature i
    let mut columns: Vec<Vec<f32>> = vec![Vec::with_capacity(num_rows); num_features];
    for row in feature_vectors {
        for (i, &val) in row.iter().enumerate() {
            columns[i].push(val);
        }
    }

    // Build Arrow arrays
    let arrays: Vec<ArrayRef> = columns
        .iter()
        .enumerate()
        .map(|(_i, col)| Arc::new(Float32Array::from(col.clone())) as ArrayRef)
        .collect();

    // Build schema
    let fields: Vec<Field> = (0..num_features)
        .map(|i| Field::new(&format!("feature_{}", i), DataType::Float32, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));

    // Build record batch
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;

    // Write to Parquet
    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

pub fn save_positions_to_parquet(
    positions: &[i32],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use arrow::array::Int32Array;

    let array = Arc::new(Int32Array::from(positions.to_vec())) as ArrayRef;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "position",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![array])?;
    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}
