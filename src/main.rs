//! Chessformer policy/value acceptance path in Rust on Axis.
mod data;
mod model;

use axis::prelude::*;
use data::PositionSample;
use model::{ChessAxes, Chessformer};
use std::{env, time::Instant};

fn tensors(
    samples: &[PositionSample],
    axes: &ChessAxes,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    if samples.is_empty() {
        return Err("cannot tensorize an empty chess batch".into());
    }
    let channels = samples[0].input.len() / 64;
    if samples
        .iter()
        .any(|sample| sample.input.len() != 64 * channels)
    {
        return Err("chess samples disagree on input channel count".into());
    }
    let mut inputs = Vec::with_capacity(samples.len() * 64 * channels);
    let mut policy = vec![0.0; samples.len() * 4096];
    let mut value = vec![0.0; samples.len() * 3];
    for (batch, sample) in samples.iter().enumerate() {
        inputs.extend_from_slice(&sample.input);
        policy[batch * 4096 + sample.move_index] = 1.0;
        value[batch * 3 + sample.outcome] = 1.0;
    }
    Ok((
        Tensor::from_slice(
            &inputs,
            [
                axes.batch.of(samples.len()),
                axes.square.of(64),
                axes.channel.of(channels),
            ],
            device,
        )?,
        Tensor::from_slice(
            &policy,
            [axes.batch.of(samples.len()), axes.move_class.of(4096)],
            device,
        )?,
        Tensor::from_slice(
            &value,
            [axes.batch.of(samples.len()), axes.outcome.of(3)],
            device,
        )?,
    ))
}

fn losses(
    policy_logits: &Tensor,
    value_logits: &Tensor,
    policy_targets: &Tensor,
    value_targets: &Tensor,
    axes: &ChessAxes,
) -> Result<(Tensor, Tensor, Tensor)> {
    let policy = policy_logits
        .categorical_cross_entropy_with_logits(policy_targets, axes.move_class)?
        .mean(axes.batch)?;
    let value = value_logits
        .categorical_cross_entropy_with_logits(value_targets, axes.outcome)?
        .mean(axes.batch)?;
    let total = policy.add(&value.scale(0.1)?)?;
    Ok((total, policy, value))
}

struct Metrics {
    total_loss: f32,
    policy_loss: f32,
    value_loss: f32,
    policy_accuracy: f32,
    value_accuracy: f32,
}

fn metrics(model: &Chessformer, samples: &[PositionSample], device: &Device) -> Result<Metrics> {
    let (inputs, policy_targets, value_targets) = tensors(samples, model.axes(), device)?;
    let (policy, value) = model.outputs(&inputs)?;
    let (total_loss, policy_loss, value_loss) = losses(
        &policy,
        &value,
        &policy_targets,
        &value_targets,
        model.axes(),
    )?;
    let policy_values = policy.to_vec()?;
    let value_values = value.to_vec()?;
    let mut policy_correct = 0;
    let mut value_correct = 0;
    for (batch, sample) in samples.iter().enumerate() {
        let policy_start = batch * 4096;
        let predicted_move = policy_values[policy_start..policy_start + 4096]
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .expect("nonempty move classes");
        policy_correct += usize::from(predicted_move == sample.move_index);
        let value_start = batch * 3;
        let predicted_value = value_values[value_start..value_start + 3]
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .expect("nonempty value classes");
        value_correct += usize::from(predicted_value == sample.outcome);
    }
    Ok(Metrics {
        total_loss: total_loss.item()?,
        policy_loss: policy_loss.item()?,
        value_loss: value_loss.item()?,
        policy_accuracy: policy_correct as f32 / samples.len() as f32,
        value_accuracy: value_correct as f32 / samples.len() as f32,
    })
}

fn print_metrics(label: &str, metrics: &Metrics) {
    println!(
        "{label} total_loss={:.6} policy_loss={:.6} value_loss={:.6} policy_accuracy={:.2}% value_accuracy={:.2}%",
        metrics.total_loss,
        metrics.policy_loss,
        metrics.value_loss,
        metrics.policy_accuracy * 100.0,
        metrics.value_accuracy * 100.0,
    );
}

fn main() -> Result<()> {
    let mut history = 7_usize;
    let mut embedding = 24_usize;
    let mut heads = 4_usize;
    let mut layers = 2_usize;
    let mut templates = 8_usize;
    let mut batch_size = 8_usize;
    let mut passes = 3_usize;
    let mut learning_rate = 3e-4_f32;
    let mut weight_decay = 1e-6_f32;
    let mut smoke = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke" => {
                smoke = true;
                history = 1;
                embedding = 12;
                heads = 3;
                layers = 1;
                templates = 2;
                batch_size = 2;
                passes = 1;
                learning_rate = 1e-3;
            }
            "--history" => history = args.next().ok_or("--history needs a value")?.parse()?,
            "--embedding" => embedding = args.next().ok_or("--embedding needs a value")?.parse()?,
            "--heads" => heads = args.next().ok_or("--heads needs a value")?.parse()?,
            "--layers" => layers = args.next().ok_or("--layers needs a value")?.parse()?,
            "--templates" => templates = args.next().ok_or("--templates needs a value")?.parse()?,
            "--batch" => batch_size = args.next().ok_or("--batch needs a value")?.parse()?,
            "--passes" => passes = args.next().ok_or("--passes needs a value")?.parse()?,
            "--learning-rate" => {
                learning_rate = args
                    .next()
                    .ok_or("--learning-rate needs a value")?
                    .parse()?
            }
            "--weight-decay" => {
                weight_decay = args.next().ok_or("--weight-decay needs a value")?.parse()?
            }
            "--help" | "-h" => {
                println!(
                    "chess-transformer [--smoke] [--history N] [--embedding N] [--heads N] \
                     [--layers N] [--templates N] [--batch N] [--passes N] \
                     [--learning-rate F] [--weight-decay F]"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    if batch_size == 0 || passes == 0 {
        return Err("batch size and passes must be positive".into());
    }

    let corpus = data::load(history)?;
    let training = if smoke {
        corpus.train[..4.min(corpus.train.len())].to_vec()
    } else {
        corpus.train.clone()
    };
    let evaluation = if smoke {
        corpus.evaluation[..2.min(corpus.evaluation.len())].to_vec()
    } else {
        corpus.evaluation.clone()
    };
    println!(
        "corpus train_games={} eval_games={} train_positions={} eval_positions={} history={} smoke_subset={}",
        corpus.train_games,
        corpus.evaluation_games,
        training.len(),
        evaluation.len(),
        corpus.history,
        smoke,
    );

    let device = Device::cuda(0)?;
    let mut model = Chessformer::new(history, embedding, heads, layers, templates)?;
    model.build(
        &Shape::new([
            model.axes().batch.of(batch_size),
            model.axes().square.of(64),
            model.axes().channel.of(12 * (history + 1)),
        ])?,
        &device,
        42,
    )?;
    println!(
        "model chessformer-gab layers={layers} embedding={embedding} heads={heads} templates={templates} parameters={}",
        model.parameter_count()
    );

    let evaluation_game_identities = evaluation
        .iter()
        .map(|sample| sample.game_identity.clone())
        .collect::<Vec<_>>();
    let training_ids = training.iter().map(|sample| sample.id).collect::<Vec<_>>();
    let mut disjoint = Disjointness::new(
        IdentityScheme::new(
            "chess-move-sequence",
            "1",
            "complete whitespace-normalized UCI move sequence for one game",
        )?,
        [
            PopulationSpec::streaming("training"),
            PopulationSpec::retained("evaluation"),
        ],
    )?;
    disjoint.observe("evaluation", evaluation_game_identities)?;
    let initial = metrics(&model, &evaluation, &device)?;
    print_metrics("initial", &initial);

    let mut loader =
        FinitePassesLoader::with_ids(training, training_ids, batch_size, passes, 0xc4e5_5f0f)?;
    let mut trainer = Trainer::new(AdamW::new(learning_rate, weight_decay)?);
    let started = Instant::now();
    while let Some(batch) = loader.next_batch()? {
        disjoint.observe(
            "training",
            batch
                .samples
                .iter()
                .map(|sample| sample.game_identity.clone()),
        )?;
        let (inputs, policy_targets, value_targets) =
            tensors(&batch.samples, model.axes(), &device)?;
        let report = trainer.step(&mut model, |model| {
            let (policy, value) = model.outputs(&inputs)?;
            Ok(losses(
                &policy,
                &value,
                &policy_targets,
                &value_targets,
                model.axes(),
            )?
            .0)
        })?;
        println!(
            "step={} samples={} pre_update_loss={:.6}",
            report.step(),
            loader.samples_delivered(),
            report.pre_update_loss()?
        );
    }
    let final_metrics = metrics(&model, &evaluation, &device)?;
    print_metrics("final", &final_metrics);
    println!("elapsed_s={:.2}", started.elapsed().as_secs_f64());
    println!("{}", loader.receipt());
    println!("{}", disjoint.assert_disjoint()?);
    if !final_metrics.total_loss.is_finite() {
        return Err("Chessformer acceptance produced a non-finite evaluation loss".into());
    }
    println!("PASS: Chessformer policy/value path completed on game-disjoint opening lines");
    Ok(())
}
