use noodles::sam::alignment::record::cigar::op::Kind as CigarKind;
use noodles::sam::alignment::record::cigar::Op;

#[derive(Debug)]
pub struct AugmentedCigar {
    pub op: CigarKind,
    pub len: usize,
    pub ref_pos: i32,
    pub read_pos: u32
}

pub struct AugmentedCigarIter<'a> {
    cigars: Box<dyn  Iterator<Item = Result<Op, std::io::Error>> + 'a>,
    ref_pos: i32,
    read_pos: u32
}

impl<'a> AugmentedCigarIter<'a> {
    pub fn new(cigars: Box<dyn  Iterator<Item = Result<Op, std::io::Error>> + 'a>, ref_pos: i32) -> Self {
        Self {
            cigars,
            ref_pos,
            read_pos: 0,
        }
    }
}

impl Iterator for AugmentedCigarIter<'_> {
    type Item = Result<AugmentedCigar, std::io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.cigars.next() {
            Some(op) => {
                match op {
                    Ok(op) => {
                        let len = op.len();
                        let ref_pos = self.ref_pos;
                        let read_pos = self.read_pos;
                        let aug = AugmentedCigar {
                            op: op.kind(),
                            len,
                            ref_pos,
                            read_pos
                        };
                        match op.kind() {
                            CigarKind::Match => {
                                self.ref_pos += op.len() as i32;
                                self.read_pos += op.len() as u32;
                            },
                            CigarKind::Insertion => {
                                self.ref_pos += 0;
                                self.read_pos += op.len() as u32;
                            },
                            CigarKind::Deletion => {
                                self.ref_pos += op.len() as i32;
                                self.read_pos += 0;
                            },
                            CigarKind::Skip => {
                                self.ref_pos += 0;
                                self.read_pos += 0;
                            },
                            CigarKind::SoftClip => {
                                self.ref_pos += 0;
                                self.read_pos += op.len() as u32;
                            },
                            CigarKind::HardClip => {
                                self.ref_pos += 0;
                                self.read_pos += 0;
                            },
                            CigarKind::Pad => {
                                self.ref_pos += 0;
                                self.read_pos += 0;
                            },
                            CigarKind::SequenceMatch => {
                                self.ref_pos += op.len() as i32;
                                self.read_pos += op.len() as u32;
                            },
                            CigarKind::SequenceMismatch => {
                                self.ref_pos += op.len() as i32;
                                self.read_pos += op.len() as u32;
                            },
                        }
                        Some(Ok(aug))
                    }
                    Err(err) => Some(Err(err))
                }
            },
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noodles::sam::alignment::record::cigar::Op;
    use noodles::sam::alignment::record::cigar::op::Kind as CigarKind;

    fn make_ops() -> Vec<Result<Op, std::io::Error>> {
        vec![
            Ok(Op::new(CigarKind::Match, 5)),
            Ok(Op::new(CigarKind::Insertion, 2)),
            Ok(Op::new(CigarKind::Deletion, 3)),
        ]
    }

    #[test]
    fn test_augmented_cigar_iter_positions() {
        let ops = make_ops();
        let mut iter = AugmentedCigarIter::new(Box::new(ops.into_iter()), 100);

        let first = iter.next().unwrap().unwrap();
        assert_eq!(first.op, CigarKind::Match);
        assert_eq!(first.len, 5);
        assert_eq!(first.ref_pos, 100);
        assert_eq!(first.read_pos, 0);

        let second = iter.next().unwrap().unwrap();
        assert_eq!(second.op, CigarKind::Insertion);
        assert_eq!(second.len, 2);
        assert_eq!(second.ref_pos, 105); // after match
        assert_eq!(second.read_pos, 5);

        let third = iter.next().unwrap().unwrap();
        assert_eq!(third.op, CigarKind::Deletion);
        assert_eq!(third.len, 3);
        assert_eq!(third.ref_pos, 105); // insertion doesn't advance ref
        assert_eq!(third.read_pos, 7); // after insertion
    }

    #[test]
    fn test_augmented_cigar_iter_none() {
        let ops: Vec<Result<Op, std::io::Error>> = vec![];
        let mut iter = AugmentedCigarIter::new(Box::new(ops.into_iter()), 0);
        assert!(iter.next().is_none());
    }
}
