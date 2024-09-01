use border_derive::BatchBase;
use border_tch_agent::TensorBatch;

#[allow(dead_code)]
#[derive(Clone, BatchBase)]
pub struct ObsBatch(TensorBatch);

fn main() {}
