use axis::prelude::Result;
use std::sync::Arc;

const CORPUS: &str = include_str!("../data/smoke-games.tsv");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Split {
    Train,
    Evaluation,
}

#[derive(Clone)]
pub struct PositionSample {
    pub id: u128,
    pub game_identity: Arc<str>,
    pub input: Vec<f32>,
    pub move_index: usize,
    pub outcome: usize,
}

pub struct Corpus {
    pub train: Vec<PositionSample>,
    pub evaluation: Vec<PositionSample>,
    pub history: usize,
    pub train_games: usize,
    pub evaluation_games: usize,
}

#[derive(Clone, Copy)]
enum ResultLabel {
    WhiteWin,
    Draw,
    BlackWin,
}

impl ResultLabel {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "1-0" => Ok(Self::WhiteWin),
            "1/2-1/2" => Ok(Self::Draw),
            "0-1" => Ok(Self::BlackWin),
            _ => Err(format!("unknown game result {value:?}").into()),
        }
    }

    fn outcome_for_side_to_move(self, white_to_move: bool) -> usize {
        match (self, white_to_move) {
            (Self::Draw, _) => 1,
            (Self::WhiteWin, true) | (Self::BlackWin, false) => 2,
            (Self::WhiteWin, false) | (Self::BlackWin, true) => 0,
        }
    }
}

#[derive(Clone)]
struct Board([u8; 64]);

impl Board {
    fn initial() -> Self {
        let mut board = [0; 64];
        board[..8].copy_from_slice(&[4, 2, 3, 5, 6, 3, 2, 4]);
        board[8..16].fill(1);
        board[48..56].fill(7);
        board[56..].copy_from_slice(&[10, 8, 9, 11, 12, 9, 8, 10]);
        Self(board)
    }

    fn apply(&mut self, from: usize, to: usize, white_to_move: bool) -> Result<()> {
        let piece = self.0[from];
        if piece == 0 {
            return Err(format!("opening line moves from empty square {from}").into());
        }
        let is_white = piece <= 6;
        if is_white != white_to_move {
            return Err(format!("opening line moves the wrong color from square {from}").into());
        }
        let target = self.0[to];
        if target != 0 && (target <= 6) == is_white {
            return Err(format!("opening line captures its own piece on square {to}").into());
        }
        self.0[from] = 0;
        self.0[to] = piece;
        Ok(())
    }

    fn normalized(&self, white_to_move: bool) -> [u8; 64] {
        if white_to_move {
            return self.0;
        }
        let mut output = [0; 64];
        for (square, &piece) in self.0.iter().enumerate() {
            let piece = match piece {
                1..=6 => piece + 6,
                7..=12 => piece - 6,
                _ => 0,
            };
            output[square ^ 56] = piece;
        }
        output
    }
}

fn square(value: &str) -> Result<usize> {
    let bytes = value.as_bytes();
    if bytes.len() != 2 || !(b'a'..=b'h').contains(&bytes[0]) || !(b'1'..=b'8').contains(&bytes[1])
    {
        return Err(format!("invalid UCI square {value:?}").into());
    }
    Ok(usize::from(bytes[1] - b'1') * 8 + usize::from(bytes[0] - b'a'))
}

fn chess_move(value: &str) -> Result<(usize, usize)> {
    if value.len() != 4 {
        return Err(format!(
            "smoke corpus currently accepts ordinary four-byte UCI moves, got {value:?}"
        )
        .into());
    }
    Ok((square(&value[..2])?, square(&value[2..])?))
}

fn encode_history(states: &[Board], ply: usize, history: usize) -> Vec<f32> {
    let channels = 12 * (history + 1);
    let mut input = vec![0.0; 64 * channels];
    for offset in 0..=history {
        let state_ply = ply.saturating_sub(offset);
        let board = states[state_ply].normalized(state_ply.is_multiple_of(2));
        for (square, &piece) in board.iter().enumerate() {
            if piece != 0 {
                input[square * channels + offset * 12 + usize::from(piece - 1)] = 1.0;
            }
        }
    }
    input
}

fn namespace(split: Split) -> u128 {
    match split {
        Split::Train => 0x0074_7261_696e,
        Split::Evaluation => 0x0000_6576_616c,
    }
}

pub fn load(history: usize) -> Result<Corpus> {
    let mut train = Vec::new();
    let mut evaluation = Vec::new();
    let mut train_games = 0;
    let mut evaluation_games = 0;
    for (line_index, raw) in CORPUS.lines().enumerate() {
        let raw = raw.trim();
        if raw.is_empty() || raw.starts_with('#') {
            continue;
        }
        let fields = raw.split('\t').collect::<Vec<_>>();
        if fields.len() != 3 {
            return Err(format!(
                "smoke corpus line {} needs three tab-separated fields",
                line_index + 1
            )
            .into());
        }
        let split = match fields[0] {
            "train" => Split::Train,
            "eval" => Split::Evaluation,
            value => return Err(format!("unknown split {value:?}").into()),
        };
        let result = ResultLabel::parse(fields[1])?;
        let moves = fields[2]
            .split_ascii_whitespace()
            .map(chess_move)
            .collect::<Result<Vec<_>>>()?;
        let game_identity: Arc<str> = Arc::from(fields[2]);
        if moves.is_empty() {
            return Err("opening line contains no moves".into());
        }
        let mut board = Board::initial();
        let mut states = Vec::with_capacity(moves.len());
        for &(from, to) in &moves {
            states.push(board.clone());
            let ply = states.len() - 1;
            board.apply(from, to, ply.is_multiple_of(2))?;
        }
        let game_index = match split {
            Split::Train => {
                let index = train_games;
                train_games += 1;
                index
            }
            Split::Evaluation => {
                let index = evaluation_games;
                evaluation_games += 1;
                index
            }
        };
        for (ply, &(from, to)) in moves.iter().enumerate() {
            let white_to_move = ply.is_multiple_of(2);
            let (from, to) = if white_to_move {
                (from, to)
            } else {
                (from ^ 56, to ^ 56)
            };
            let sample = PositionSample {
                id: (namespace(split) << 64) | ((game_index as u128) << 32) | ply as u128,
                game_identity: Arc::clone(&game_identity),
                input: encode_history(&states, ply, history),
                move_index: from * 64 + to,
                outcome: result.outcome_for_side_to_move(white_to_move),
            };
            match split {
                Split::Train => train.push(sample),
                Split::Evaluation => evaluation.push(sample),
            }
        }
    }
    if train.is_empty() || evaluation.is_empty() {
        return Err("smoke corpus needs nonempty train and evaluation splits".into());
    }
    Ok(Corpus {
        train,
        evaluation,
        history,
        train_games,
        evaluation_games,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_game_disjoint_and_well_formed() -> Result<()> {
        let corpus = load(7)?;
        assert_eq!(corpus.train_games, 6);
        assert_eq!(corpus.evaluation_games, 2);
        assert!(
            corpus
                .train
                .iter()
                .all(|sample| sample.input.len() == 64 * 96)
        );
        assert!(corpus.train.iter().all(|sample| {
            !corpus
                .evaluation
                .iter()
                .any(|other| sample.game_identity == other.game_identity)
        }));
        Ok(())
    }

    #[test]
    fn black_move_is_mirrored_to_active_side() -> Result<()> {
        let corpus = load(0)?;
        let black_e7e5 = &corpus.train[1];
        let e2 = square("e2")?;
        let e4 = square("e4")?;
        assert_eq!(black_e7e5.move_index, e2 * 64 + e4);
        assert_eq!(black_e7e5.input[channel_index(e2, 0)], 1.0);
        Ok(())
    }

    fn channel_index(square: usize, piece_channel: usize) -> usize {
        square * 12 + piece_channel
    }
}
