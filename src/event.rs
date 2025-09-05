use noodles::sam::alignment::record::cigar::op::Kind as CigarKind;
use crate::extract::cigar::AugmentedCigar;


#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum EventKind {
    LeftClip = 0,
    RightClip = 1,
    MatchStart = 2,
    MatchStop = 3,
    InsertStart = 4,
    InsertStop = 5,
    DeleteStart = 6,
    DeleteStop = 7,
}

impl EventKind {
    pub const MAX: usize = EventKind::DeleteStop as usize;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub kind: EventKind,
    pub position: i32,
    pub len: usize,
    pub support: usize,
}

impl Event {
    pub fn new(kind: EventKind, position: i32, len: usize, support: usize) -> Self {
        Self { kind, position, len, support }
    }

    pub fn to_feature_vector(&self, offset: i32) -> Vec<f32> {
        let mut res = vec![0.0; EventKind::MAX + 3];
        res[self.kind as usize] = 1.0;
        res[EventKind::MAX + 0] = (self.position - offset) as f32;
        res[EventKind::MAX + 1] = self.len as f32;
        res[EventKind::MAX + 2] = self.support as f32;
        res
    }

    pub fn events(aug: AugmentedCigar) -> Vec<Event> {
        let mut events = Vec::new();
        match aug.op {
            CigarKind::Match => {
                events.push(Event::new(EventKind::MatchStart, aug.ref_pos, aug.len, 1));
                events.push(Event::new(EventKind::MatchStop, aug.ref_pos + aug.len as i32 - 1, aug.len, 1));
            },
            CigarKind::Insertion => {
                events.push(Event::new(EventKind::InsertStart, aug.ref_pos, aug.len, 1));
                events.push(Event::new(EventKind::InsertStop, aug.ref_pos + aug.len as i32 - 1, aug.len, 1));
            },
            CigarKind::Deletion => {
                events.push(Event::new(EventKind::DeleteStart, aug.ref_pos, aug.len, 1));
                events.push(Event::new(EventKind::DeleteStop, aug.ref_pos + aug.len as i32 - 1, aug.len, 1));
            },
            CigarKind::SoftClip => {
                if aug.read_pos == 0 {
                    events.push(Event::new(EventKind::LeftClip, aug.ref_pos, aug.len, 1));
                } else {
                    events.push(Event::new(EventKind::RightClip, aug.ref_pos + aug.len as i32, aug.len, 1));
                }
            },
            _ => {}
        }
        events
    }
}

impl Default for Event {
    fn default() -> Self {
        Self {
            kind: EventKind::LeftClip,
            position: 0,
            len: 0,
            support: 0,
        }
    }
}