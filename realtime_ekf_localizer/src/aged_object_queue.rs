//! Port of `aged_object_queue.hpp` — the FIFO measurement queue whose objects are re-enqueued
//! (aged) until they have been consumed `max_age` times, implementing measurement smoothing.

use alloc::collections::VecDeque;

/// FIFO queue with per-object age (port of `AgedObjectQueue<Object>`).
///
/// The C++ `pop`/`pop_increment_age`/`back` on an empty queue are undefined behavior; the port
/// returns `Option` instead.
#[derive(Clone, Debug)]
pub struct AgedObjectQueue<T> {
    max_age: usize,
    max_queue_size: usize,
    objects: VecDeque<T>,
    ages: VecDeque<usize>,
}

impl<T: Clone> AgedObjectQueue<T> {
    /// Create a queue. `max_queue_size` values not larger than `max_age` collapse to
    /// `max_age`, mirroring the C++ constructor default.
    #[must_use]
    pub fn new(max_age: usize, max_queue_size: usize) -> Self {
        Self {
            max_age,
            max_queue_size: if max_queue_size > max_age {
                max_queue_size
            } else {
                max_age
            },
            objects: VecDeque::new(),
            ages: VecDeque::new(),
        }
    }

    /// `true` when the queue holds no objects.
    #[must_use]
    pub fn empty(&self) -> bool {
        self.size() == 0
    }

    /// `true` when the queue size exceeds `max_queue_size` (checked after `push`).
    #[must_use]
    pub fn exceeded(&self) -> bool {
        self.size() > self.max_queue_size
    }

    /// Number of stored objects.
    #[must_use]
    pub fn size(&self) -> usize {
        self.objects.len()
    }

    /// The configured maximum age.
    #[must_use]
    pub fn max_age(&self) -> usize {
        self.max_age
    }

    /// The configured maximum queue size.
    #[must_use]
    pub fn max_queue_size(&self) -> usize {
        self.max_queue_size
    }

    /// Most recently pushed object (C++ `back`; `None` when empty).
    #[must_use]
    pub fn back(&self) -> Option<&T> {
        self.objects.back()
    }

    /// Append an object with age 0.
    pub fn push(&mut self, object: T) {
        self.objects.push_back(object);
        self.ages.push_back(0);
    }

    /// Remove and return the front object (C++ `pop`; `None` when empty).
    pub fn pop(&mut self) -> Option<T> {
        let object = self.objects.pop_front();
        let _age: Option<usize> = self.ages.pop_front();
        object
    }

    /// Remove the front object, increment its age (saturating: ages are bounded by `max_age`
    /// in practice), and re-enqueue it unless the new age reached `max_age`. Returns the
    /// object (`None` when empty).
    pub fn pop_increment_age(&mut self) -> Option<T> {
        let object = self.objects.pop_front()?;
        let age = self.ages.pop_front()?.saturating_add(1);

        if age < self.max_age {
            self.objects.push_back(object.clone());
            self.ages.push_back(age);
        }

        Some(object)
    }

    /// Drop all objects.
    pub fn clear(&mut self) {
        self.objects.clear();
        self.ages.clear();
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::float_cmp,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;
    use alloc::string::String;

    // Transcription of test_aged_object_queue.cpp.

    #[test]
    fn discards_object_when_age_reaches_maximum() {
        let mut queue: AgedObjectQueue<String> = AgedObjectQueue::new(3, 0);

        queue.push(String::from("a"));
        assert_eq!(queue.size(), 1);

        queue.pop_increment_age(); // age = 1
        assert_eq!(queue.size(), 1);

        queue.pop_increment_age(); // age = 2
        assert_eq!(queue.size(), 1);

        queue.pop_increment_age(); // age = 3
        assert_eq!(queue.size(), 0);
    }

    #[test]
    fn multiple_objects() {
        let mut queue: AgedObjectQueue<String> = AgedObjectQueue::new(3, 0);

        queue.push(String::from("a"));
        assert_eq!(queue.size(), 1);
        assert_eq!(queue.pop_increment_age().unwrap(), "a"); // age of a = 1
        assert_eq!(queue.size(), 1);
        assert_eq!(queue.pop_increment_age().unwrap(), "a"); // age of a = 2

        queue.push(String::from("b"));

        assert_eq!(queue.pop_increment_age().unwrap(), "a"); // age of a = 3
        assert_eq!(queue.size(), 1);

        assert_eq!(queue.pop_increment_age().unwrap(), "b"); // age of b = 1
        assert_eq!(queue.size(), 1);

        assert_eq!(queue.pop_increment_age().unwrap(), "b"); // age of b = 2
        assert_eq!(queue.size(), 1);

        assert_eq!(queue.pop_increment_age().unwrap(), "b"); // age of b = 3
        assert_eq!(queue.size(), 0);
    }

    #[test]
    fn empty() {
        let mut queue: AgedObjectQueue<String> = AgedObjectQueue::new(2, 0);

        assert!(queue.empty());
        queue.push(String::from("a"));
        assert!(!queue.empty());
        queue.pop_increment_age();
        queue.pop_increment_age();
        assert!(queue.empty());

        // Port hardening: empty pops are None, not UB.
        assert!(queue.pop().is_none());
        assert!(queue.pop_increment_age().is_none());
        assert!(queue.back().is_none());
    }

    #[test]
    fn clear() {
        let mut queue: AgedObjectQueue<String> = AgedObjectQueue::new(3, 0);
        queue.push(String::from("a"));
        queue.push(String::from("b"));
        assert_eq!(queue.size(), 2);
        queue.clear();
        assert_eq!(queue.size(), 0);
    }

    #[test]
    fn back() {
        let mut queue: AgedObjectQueue<String> = AgedObjectQueue::new(3, 0);
        queue.push(String::from("a"));
        assert_eq!(queue.back().unwrap(), "a");
        queue.push(String::from("b"));
        assert_eq!(queue.back().unwrap(), "b");
    }

    #[test]
    fn max_queue_size_collapses_to_max_age() {
        let queue: AgedObjectQueue<String> = AgedObjectQueue::new(5, 2);
        assert_eq!(queue.max_queue_size(), 5);
        assert_eq!(queue.max_age(), 5);
        let queue: AgedObjectQueue<String> = AgedObjectQueue::new(2, 5);
        assert_eq!(queue.max_queue_size(), 5);
    }
}
