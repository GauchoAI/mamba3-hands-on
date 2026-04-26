//! PTX Mamba-3 engine — inference + (future) training on NVIDIA GPUs via
//! hand-written CUDA C kernels compiled to PTX with strict FP32 precision.

pub mod batch_format;
pub mod model;
pub mod runtime;
pub mod scheduler;
pub mod scratch;
pub mod train_scratch;
pub mod trainer;

pub use model::PtxModel;
pub use runtime::PtxContext;
pub use scheduler::{Job, JobRunner, Loss, Optimizer, Schedule, Scheduler, SchedulerEvent, Stage};
pub use scratch::PtxScratch;
pub use train_scratch::TrainScratch;
pub use trainer::{KdInputs, LossModifier, PtxTrainer};
