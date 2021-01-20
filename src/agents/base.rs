pub enum OptInterval {
    Steps(usize),
    Episodes(usize)
}

impl OptInterval {
    pub fn counter(self) -> OptIntervalCounter {
        OptIntervalCounter {
            opt_interval: self,
            count: 0
        }
    }
}

pub struct OptIntervalCounter {
    opt_interval: OptInterval,
    count: usize
}

impl OptIntervalCounter {
    pub fn do_optimize(&mut self, is_done: &[f32]) -> bool {
        let is_done_any = is_done.iter().fold(0, |x, v| x + *v as i32) > 0;
        match self.opt_interval {
            OptInterval::Steps(interval) => {
                self.count += 1;
                if self.count == interval {
                    self.count = 0;
                    true
                }
                else {
                    false
                }
            },
            OptInterval::Episodes(interval) => {
                if is_done_any {
                    self.count += 1;
                    if self.count == interval {
                        self.count = 0;
                        true
                    }
                    else {
                        false
                    }
                }
                else {
                    false
                }
            }
        }
    }
}
