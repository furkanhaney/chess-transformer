use axis::prelude::*;

pub struct ChessAxes {
    pub batch: Axis,
    pub square: Axis,
    pub channel: Axis,
    pub feature: Axis,
    pub move_class: Axis,
    pub outcome: Axis,
    head: Dim,
    head_feature: Dim,
    feed_forward: Axis,
    query_square: Axis,
    key_square: Axis,
    source_square: Axis,
    destination_square: Axis,
    gab_hidden: Axis,
    gab_weight: Axis,
    template: Dim,
    square_pair: Axis,
}

impl ChessAxes {
    pub fn new(embedding: usize, heads: usize, templates: usize) -> Result<Self> {
        if embedding == 0 || heads == 0 || templates == 0 || !embedding.is_multiple_of(heads) {
            return Err(
                "embedding must be positive and divisible by heads; templates must be positive"
                    .into(),
            );
        }
        let square = Axis::new("square");
        Ok(Self {
            batch: Axis::new("batch"),
            square,
            channel: Axis::new("piece_history_channel"),
            feature: Axis::new("feature"),
            move_class: Axis::new("move"),
            outcome: Axis::new("outcome"),
            head: Axis::new("head").of(heads),
            head_feature: Axis::new("head_feature").of(embedding / heads),
            feed_forward: Axis::new("feed_forward"),
            query_square: square.role("query_square"),
            key_square: square.role("key_square"),
            source_square: square.role("source_square"),
            destination_square: square.role("destination_square"),
            gab_hidden: Axis::new("gab_hidden"),
            gab_weight: Axis::new("gab_head_template"),
            template: Axis::new("gab_template").of(templates),
            square_pair: Axis::new("square_pair"),
        })
    }

    pub fn embedding(&self) -> usize {
        self.head.extent * self.head_feature.extent
    }
}

struct GeometricBias {
    square: Axis,
    head: Dim,
    template: Dim,
    pair: Axis,
    query: Axis,
    key: Axis,
    gab_weight: Axis,
    compress: Linear,
    norm1: LayerNorm,
    mix: Linear,
    norm2: LayerNorm,
}

impl GeometricBias {
    fn new(axes: &ChessAxes) -> Result<Self> {
        let hidden = (axes.embedding() / 2).max(4);
        Ok(Self {
            square: axes.square,
            head: axes.head,
            template: axes.template,
            pair: axes.square_pair,
            query: axes.query_square,
            key: axes.key_square,
            gab_weight: axes.gab_weight,
            compress: Linear::new(axes.feature, axes.gab_hidden.of(hidden)),
            norm1: LayerNorm::new(axes.gab_hidden, 1e-5)?,
            mix: Linear::new(
                axes.gab_hidden,
                axes.gab_weight.of(axes.head.extent * axes.template.extent),
            ),
            norm2: LayerNorm::new(axes.gab_weight, 1e-5)?,
        })
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<()> {
        let pooled = Shape::new(
            input
                .dims()
                .iter()
                .copied()
                .filter(|dim| dim.axis != self.square),
        )?;
        let hidden = self.compress.build(&pooled, device, seed)?;
        self.norm1.build(&hidden, device, seed.wrapping_add(1))?;
        let weights = self.mix.build(&hidden, device, seed.wrapping_add(2))?;
        self.norm2.build(&weights, device, seed.wrapping_add(3))?;
        Ok(())
    }

    fn forward(&self, input: &Tensor, templates: &Parameter) -> Result<Tensor> {
        self.norm2
            .forward(
                &self
                    .mix
                    .forward(
                        &self
                            .norm1
                            .forward(&self.compress.forward(&input.mean(self.square)?)?.gelu()?)?,
                    )?
                    .gelu()?,
            )?
            .split(self.gab_weight, [self.head, self.template])?
            .contract(&templates.tensor(), self.template.axis)?
            .split(self.pair, [self.query.of(64), self.key.of(64)])
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        [
            ("compress", &self.compress as &dyn Module),
            ("norm1", &self.norm1),
            ("mix", &self.mix),
            ("norm2", &self.norm2),
        ]
        .into_iter()
        .flat_map(|(prefix, module)| {
            module
                .named_parameters()
                .into_iter()
                .map(move |(name, parameter)| (format!("{prefix}.{name}"), parameter))
        })
        .collect()
    }
}

struct GabAttention {
    square: Axis,
    feature: Axis,
    head: Dim,
    head_feature: Dim,
    query_square: Axis,
    key_square: Axis,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    gab: GeometricBias,
}

impl GabAttention {
    fn new(axes: &ChessAxes) -> Result<Self> {
        let projection = || Linear::new(axes.feature, axes.feature.of(axes.embedding()));
        Ok(Self {
            square: axes.square,
            feature: axes.feature,
            head: axes.head,
            head_feature: axes.head_feature,
            query_square: axes.query_square,
            key_square: axes.key_square,
            query: projection(),
            key: projection(),
            value: projection(),
            output: projection(),
            gab: GeometricBias::new(axes)?,
        })
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        input.extent(self.square)?;
        let projected = self.query.build(input, device, seed)?;
        self.key.build(input, device, seed.wrapping_add(1))?;
        self.value.build(input, device, seed.wrapping_add(2))?;
        self.output
            .build(&projected, device, seed.wrapping_add(3))?;
        self.gab.build(input, device, seed.wrapping_add(4))?;
        Ok(input.clone())
    }

    fn forward(&self, input: &Tensor, templates: &Parameter) -> Result<Tensor> {
        let split = |tensor: Tensor, square| {
            tensor
                .split(self.feature, [self.head, self.head_feature])?
                .rename(self.square, square)
        };
        let query = split(self.query.forward(input)?, self.query_square)?;
        let key = split(self.key.forward(input)?, self.key_square)?;
        let value = split(self.value.forward(input)?, self.key_square)?;
        let logits = query
            .contract(&key, self.head_feature.axis)?
            .scale(1.0 / (self.head_feature.extent as f32).sqrt())?
            .add(&self.gab.forward(input, templates)?)?;
        self.output.forward(
            &logits
                .softmax(self.key_square)?
                .contract(&value, self.key_square)?
                .merge([self.head.axis, self.head_feature.axis], self.feature)?
                .rename(self.query_square, self.square)?,
        )
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut parameters = [
            ("query", &self.query as &dyn Module),
            ("key", &self.key),
            ("value", &self.value),
            ("output", &self.output),
        ]
        .into_iter()
        .flat_map(|(prefix, module)| {
            module
                .named_parameters()
                .into_iter()
                .map(move |(name, parameter)| (format!("{prefix}.{name}"), parameter))
        })
        .collect::<Vec<_>>();
        parameters.extend(
            self.gab
                .named_parameters()
                .into_iter()
                .map(|(name, parameter)| (format!("gab.{name}"), parameter)),
        );
        parameters
    }
}

struct TransformerBlock {
    norm1: LayerNorm,
    attention: GabAttention,
    norm2: LayerNorm,
    feed_forward: Sequential,
}

impl TransformerBlock {
    fn new(axes: &ChessAxes) -> Result<Self> {
        Ok(Self {
            norm1: LayerNorm::new(axes.feature, 1e-5)?,
            attention: GabAttention::new(axes)?,
            norm2: LayerNorm::new(axes.feature, 1e-5)?,
            feed_forward: Sequential::new((
                Linear::new(axes.feature, axes.feed_forward.of(2 * axes.embedding())),
                GELU,
                Linear::new(axes.feed_forward, axes.feature.of(axes.embedding())),
            )),
        })
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<()> {
        let normalized = self.norm1.build(input, device, seed)?;
        self.attention
            .build(&normalized, device, seed.wrapping_add(1))?;
        let normalized = self.norm2.build(input, device, seed.wrapping_add(2))?;
        let output = self
            .feed_forward
            .build(&normalized, device, seed.wrapping_add(3))?;
        if &output != input {
            return Err("transformer feed-forward path changed the residual shape".into());
        }
        Ok(())
    }

    fn forward(&self, input: &Tensor, templates: &Parameter) -> Result<Tensor> {
        let attended = input.add(
            &self
                .attention
                .forward(&self.norm1.forward(input)?, templates)?,
        )?;
        attended.add(&self.feed_forward.forward(&self.norm2.forward(&attended)?)?)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut parameters = [
            ("norm1", &self.norm1 as &dyn Module),
            ("norm2", &self.norm2),
            ("feed_forward", &self.feed_forward),
        ]
        .into_iter()
        .flat_map(|(prefix, module)| {
            module
                .named_parameters()
                .into_iter()
                .map(move |(name, parameter)| (format!("{prefix}.{name}"), parameter))
        })
        .collect::<Vec<_>>();
        parameters.extend(
            self.attention
                .named_parameters()
                .into_iter()
                .map(|(name, parameter)| (format!("attention.{name}"), parameter)),
        );
        parameters
    }
}

pub struct Chessformer {
    axes: ChessAxes,
    input_channels: usize,
    input: Linear,
    templates: Option<Parameter>,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    source: Linear,
    destination: Linear,
    value_norm: LayerNorm,
    value: Sequential,
}

impl Chessformer {
    pub fn new(
        history: usize,
        embedding: usize,
        heads: usize,
        layers: usize,
        templates: usize,
    ) -> Result<Self> {
        if layers == 0 {
            return Err("Chessformer needs at least one transformer layer".into());
        }
        let axes = ChessAxes::new(embedding, heads, templates)?;
        let blocks = (0..layers)
            .map(|_| TransformerBlock::new(&axes))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input_channels: 12 * (history + 1),
            input: Linear::new(axes.channel, axes.feature.of(embedding)),
            templates: None,
            final_norm: LayerNorm::new(axes.feature, 1e-5)?,
            source: Linear::new(axes.feature, axes.feature.of(embedding)),
            destination: Linear::new(axes.feature, axes.feature.of(embedding)),
            value_norm: LayerNorm::new(axes.feature, 1e-5)?,
            value: Sequential::new((
                Linear::new(axes.feature, axes.feed_forward.of(128)),
                ReLU,
                Linear::new(axes.feed_forward, axes.outcome.of(3)),
            )),
            blocks,
            axes,
        })
    }

    pub fn axes(&self) -> &ChessAxes {
        &self.axes
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters()
            .iter()
            .map(|parameter| parameter.tensor().shape().len())
            .sum()
    }

    fn encode(&self, input: &Tensor) -> Result<Tensor> {
        let templates = self
            .templates
            .as_ref()
            .ok_or("Chessformer must be built before forward")?;
        let mut encoded = self.input.forward(input)?;
        for block in &self.blocks {
            encoded = block.forward(&encoded, templates)?;
        }
        self.final_norm.forward(&encoded)
    }

    pub fn outputs(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let encoded = self.encode(input)?;
        let source = self
            .source
            .forward(&encoded)?
            .rename(self.axes.square, self.axes.source_square)?;
        let destination = self
            .destination
            .forward(&encoded)?
            .rename(self.axes.square, self.axes.destination_square)?;
        let policy = source
            .contract(&destination, self.axes.feature)?
            .scale(1.0 / (self.axes.embedding() as f32).sqrt())?
            .merge(
                [self.axes.source_square, self.axes.destination_square],
                self.axes.move_class,
            )?;
        let value = self
            .value
            .forward(&self.value_norm.forward(&encoded.mean(self.axes.square)?)?)?;
        Ok((policy, value))
    }
}

impl Module for Chessformer {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        if input.extent(self.axes.square)? != 64
            || input.extent(self.axes.channel)? != self.input_channels
        {
            return Err(format!(
                "Chessformer expects 64 squares and {} piece-history channels",
                self.input_channels
            )
            .into());
        }
        Shape::new([
            self.axes.batch.of(input.extent(self.axes.batch)?),
            self.axes.move_class.of(4096),
        ])
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let output = self.output_shape(input)?;
        let mut encoded = self.input.build(input, device, seed)?;
        if self.templates.is_none() {
            let mut state = seed.wrapping_add(0x6a09_e667_f3bc_c909).max(1);
            let count = 4096 * self.axes.template.extent;
            let values = (0..count)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 40) as f32 / (1_u32 << 24) as f32 * 2.0 - 1.0) * 0.02
                })
                .collect::<Vec<_>>();
            self.templates = Some(Parameter::new(Tensor::from_slice(
                &values,
                [self.axes.square_pair.of(4096), self.axes.template],
                device,
            )?));
        }
        for (index, block) in self.blocks.iter_mut().enumerate() {
            block.build(&encoded, device, seed.wrapping_add(100 + index as u64 * 20))?;
        }
        encoded = self
            .final_norm
            .build(&encoded, device, seed.wrapping_add(10_000))?;
        self.source
            .build(&encoded, device, seed.wrapping_add(10_001))?;
        self.destination
            .build(&encoded, device, seed.wrapping_add(10_002))?;
        let pooled = Shape::new(
            encoded
                .dims()
                .iter()
                .copied()
                .filter(|dim| dim.axis != self.axes.square),
        )?;
        let pooled = self
            .value_norm
            .build(&pooled, device, seed.wrapping_add(10_003))?;
        self.value
            .build(&pooled, device, seed.wrapping_add(10_004))?;
        Ok(output)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        Ok(self.outputs(input)?.0)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut parameters = Vec::new();
        if let Some(templates) = &self.templates {
            parameters.push(("gab_templates".into(), templates.clone()));
        }
        parameters.extend(
            self.input
                .named_parameters()
                .into_iter()
                .map(|(name, parameter)| (format!("input.{name}"), parameter)),
        );
        for (index, block) in self.blocks.iter().enumerate() {
            parameters.extend(
                block
                    .named_parameters()
                    .into_iter()
                    .map(|(name, parameter)| (format!("blocks.{index}.{name}"), parameter)),
            );
        }
        for (prefix, module) in [
            ("final_norm", &self.final_norm as &dyn Module),
            ("policy.source", &self.source),
            ("policy.destination", &self.destination),
            ("value.norm", &self.value_norm),
            ("value.head", &self.value),
        ] {
            parameters.extend(
                module
                    .named_parameters()
                    .into_iter()
                    .map(|(name, parameter)| (format!("{prefix}.{name}"), parameter)),
            );
        }
        parameters
    }
}
