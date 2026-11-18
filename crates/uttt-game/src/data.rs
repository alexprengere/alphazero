use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::schema::{BOARD_CELLS, FEATURE_SIZE};

const MAGIC: &[u8; 8] = b"UTTTSP1\n";

pub struct TrainingSample {
    pub features: [u8; FEATURE_SIZE],
    pub policy: [f32; BOARD_CELLS],
    pub value: f32,
}

pub struct SampleWriter {
    output: BufWriter<File>,
}

impl SampleWriter {
    pub fn create(path: &Path) -> io::Result<Self> {
        let mut output = BufWriter::new(File::create(path)?);
        output.write_all(MAGIC)?;
        Ok(Self { output })
    }

    pub fn write(&mut self, sample: &TrainingSample) -> io::Result<()> {
        self.output.write_all(&sample.features)?;
        for value in sample.policy {
            self.output.write_all(&value.to_le_bytes())?;
        }
        self.output.write_all(&sample.value.to_le_bytes())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}

pub fn read_samples(path: &Path) -> io::Result<Vec<TrainingSample>> {
    let mut input = BufReader::new(File::open(path)?);
    let mut magic = [0; MAGIC.len()];
    input.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid self-play file",
        ));
    }

    let mut samples = Vec::new();
    loop {
        let mut first = [0; 1];
        if input.read(&mut first)? == 0 {
            break;
        }
        let mut features = [0; FEATURE_SIZE];
        features[0] = first[0];
        input.read_exact(&mut features[1..])?;

        let mut policy = [0.0; BOARD_CELLS];
        for value in &mut policy {
            let mut bytes = [0; 4];
            input.read_exact(&mut bytes)?;
            *value = f32::from_le_bytes(bytes);
        }
        let mut value_bytes = [0; 4];
        input.read_exact(&mut value_bytes)?;
        samples.push(TrainingSample {
            features,
            policy,
            value: f32::from_le_bytes(value_bytes),
        });
    }
    Ok(samples)
}
