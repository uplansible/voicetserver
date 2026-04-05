// Custom words text correction using Aho-Corasick multi-pattern matching.
//
// File format (~/.config/voicetserver/custom_words.txt):
//   # comment line           — ignored
//   wrong=correct            — replacement pair: replaces every occurrence of "wrong" with "correct"
//   PlainTerm                — stored as-is (no correction effect yet; Phase 3 vocab boosting)
//
// Example:
//   Migration=Miktion
//   Miktion
//   Diurese

use std::path::Path;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};

pub struct WordsCorrector {
    /// Compiled automaton over all replacement patterns. None if no replacement pairs defined.
    automaton:    Option<AhoCorasick>,
    /// Replacement strings — index i corresponds to automaton pattern i.
    replacements: Vec<String>,
    /// All non-comment lines as stored in the file (returned by GET /words).
    pub raw_lines: Vec<String>,
}

impl WordsCorrector {
    /// Load from file. Returns an empty corrector if the file does not exist.
    pub fn load(path: &Path) -> Self {
        let content = std::fs::read_to_string(path).unwrap_or_default();
        Self::from_str(&content)
    }

    /// Parse the custom words format from a string.
    pub fn from_str(content: &str) -> Self {
        let mut patterns:     Vec<String> = Vec::new();
        let mut replacements: Vec<String> = Vec::new();
        let mut raw_lines:    Vec<String> = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            raw_lines.push(line.to_string());
            if let Some((pat, rep)) = line.split_once('=') {
                let pat = pat.trim().to_string();
                let rep = rep.trim().to_string();
                if !pat.is_empty() && !rep.is_empty() {
                    patterns.push(pat);
                    replacements.push(rep);
                }
            }
            // Plain terms: kept in raw_lines for the API, no automaton entry yet
        }

        let automaton = if patterns.is_empty() {
            None
        } else {
            AhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostFirst)
                .build(&patterns)
                .ok()
        };

        Self { automaton, replacements, raw_lines }
    }

    /// Apply all replacement pairs to `text` and return the corrected string.
    /// Returns the input unchanged if no replacement pairs are defined.
    pub fn apply(&self, text: &str) -> String {
        match &self.automaton {
            None     => text.to_string(),
            Some(ac) => ac.replace_all(text, &self.replacements),
        }
    }
}
