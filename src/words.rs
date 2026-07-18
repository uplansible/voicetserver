// Custom words text correction using Aho-Corasick multi-pattern matching.
//
// File format (~/.config/voicetserver/custom_words.txt):
//   # comment line           — ignored
//   wrong=correct            — replacement pair: replaces every occurrence of "wrong" with "correct"
//   PlainTerm                — fuzzy phonetic target: transcribed words that sound like it
//                              are snapped onto this canonical spelling (see FuzzyMatcher)
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

    /// Plain vocabulary terms — non-comment lines without an `=` replacement.
    /// These are used as targets for the fuzzy phonetic matcher (below).
    pub fn plain_terms(&self) -> Vec<&str> {
        self.raw_lines
            .iter()
            .filter(|l| !l.contains('='))
            .map(|l| l.as_str())
            .collect()
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

// ---------------------------------------------------------------------------
// Fuzzy phonetic hotword correction
// ---------------------------------------------------------------------------
//
// The model often transcribes unfamiliar proper names / medical terms with a
// slightly different (phonetically equivalent) spelling each time, so exact
// `wrong=correct` pairs cannot keep up. The FuzzyMatcher snaps any transcribed
// word that is *phonetically close* to a known hotword onto the canonical
// spelling, regardless of which variant the model emitted. Targets are the
// plain (non-`=`) terms in custom_words.txt.
//
// Matching is gated by BOTH a *near* Kölner Phonetik (Cologne phonetics) code
// AND a bounded normalized Levenshtein distance, to avoid replacing legitimately
// different words that merely sound similar.
//
// Voxtral has no prompt/biasing mechanism (unlike the qwen3 sister project), so
// it cannot be nudged toward the canonical spelling of an unfamiliar name. Its
// raw output therefore drifts further than an exact-code match can bridge — most
// commonly a prepended/dropped edge vowel, which the Kölner coder turns into a
// single leading `0` (e.g. "Betmiga"→`1264` vs "Epetmika"→`01264`). Requiring an
// *identical* code rejected these outright. We instead (a) ignore a leading `0`
// when comparing, and (b) allow a small Kölner-code edit distance
// (`FUZZY_MAX_CODE_DIST`). The orthographic Levenshtein gate stays the
// false-positive backstop.

/// Words shorter than this are never fuzzy-matched (too collision-prone).
const FUZZY_MIN_LEN: usize = 4;

/// Maximum Kölner-Phonetik code edit distance still treated as a phonetic match.
/// 0 (identity) plus one edit absorbs a single voicing/segment slip beyond the
/// leading-`0` normalisation. Kept small so the orthographic gate dominates.
const FUZZY_MAX_CODE_DIST: usize = 1;

/// Fuzzy matcher built from the canonical hotword/vocabulary terms
/// (the plain terms in custom_words.txt).
pub struct FuzzyMatcher {
    /// (canonical spelling, Kölner Phonetik code) for each single-word target.
    /// Multi-word / hyphenated / digit-bearing terms are excluded because the
    /// word scanner splits on non-alphabetic characters and could not match them.
    targets: Vec<(String, String)>,
}

impl FuzzyMatcher {
    /// Build a matcher from the canonical hotword terms.
    pub fn new(terms: &[String]) -> Self {
        let targets = terms
            .iter()
            .filter(|t| is_single_word(t))
            .map(|t| (t.clone(), koelner_phonetik(t)))
            .filter(|(_, code)| !code.is_empty())
            .collect();
        Self { targets }
    }

    /// Build a matcher directly from a WordsCorrector's plain terms.
    pub fn from_corrector(corrector: &WordsCorrector) -> Self {
        let terms: Vec<String> = corrector.plain_terms().iter().map(|s| s.to_string()).collect();
        Self::new(&terms)
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Snap phonetically-close words in `text` onto their canonical hotword
    /// spelling. `max_ratio` is the maximum normalized Levenshtein distance
    /// (distance / longer-word-length) accepted as a match. Word separators
    /// (spaces, punctuation) are preserved verbatim.
    pub fn correct(&self, text: &str, max_ratio: f32) -> String {
        if self.targets.is_empty() {
            return text.to_string();
        }
        let mut out = String::with_capacity(text.len());
        let mut word = String::new();
        for c in text.chars() {
            if c.is_alphabetic() {
                word.push(c);
            } else {
                if !word.is_empty() {
                    out.push_str(&self.snap(&word, max_ratio));
                    word.clear();
                }
                out.push(c);
            }
        }
        if !word.is_empty() {
            out.push_str(&self.snap(&word, max_ratio));
        }
        out
    }

    /// Decide whether a single word should be replaced by a canonical hotword.
    /// Returns the canonical spelling on a match, otherwise the word unchanged.
    fn snap(&self, word: &str, max_ratio: f32) -> String {
        if word.chars().count() < FUZZY_MIN_LEN {
            return word.to_string();
        }
        let code = koelner_phonetik(word);
        if code.is_empty() {
            return word.to_string();
        }
        let code_n = strip_leading_zero(&code);
        let wl = word.to_lowercase();

        let mut best: Option<(&str, usize)> = None;
        for (canon, ccode) in &self.targets {
            // Phonetic gate: near (not identical) codes, ignoring a leading edge-vowel `0`.
            if levenshtein(strip_leading_zero(ccode), code_n) > FUZZY_MAX_CODE_DIST {
                continue;
            }
            let cl = canon.to_lowercase();
            if cl == wl {
                // Already the canonical spelling — leave untouched.
                return word.to_string();
            }
            let d = levenshtein(&wl, &cl);
            let maxlen = wl.chars().count().max(cl.chars().count());
            if maxlen == 0 {
                continue;
            }
            if (d as f32 / maxlen as f32) <= max_ratio
                && best.map_or(true, |(_, bd)| d < bd)
            {
                best = Some((canon, d));
            }
        }
        best.map_or_else(|| word.to_string(), |(canon, _)| canon.to_string())
    }
}

// ---------------------------------------------------------------------------
// Abbreviation / acronym expansion (letter-name pass)
// ---------------------------------------------------------------------------
//
// Dictated acronyms come out of the model as German *letter names*: "M-R-I" is
// transcribed as `Em Er I` / `EM-ER-I` — separate tokens the word-by-word fuzzy
// matcher can never join, and hyphen/digit-bearing targets like `TUR-B` are
// excluded from fuzzy matching entirely. This pass handles them:
//
//   1. Targets are the plain custom_words.txt terms that *look like* acronyms:
//      2–6 letters, all uppercase, only letters/digits/hyphens (MRI, PSA, TUR-B).
//   2. The final transcript is scanned for runs of ≥2 adjacent tokens that are
//      ALL recognized German letter names (`em→M, er→R, i→I, …`; a single
//      alphabetic character stands for itself).
//   3. The run's letters are joined and compared against each target's
//      letters-only key (`TUR-B` → `TURB`). On a match the whole run is
//      replaced by the canonical spelling from custom_words.txt.
//
// Requiring every token to be a letter name AND the full run to match a known
// target keeps false positives low even though `er`/`es` are common German
// words. Longest run wins; on no match the window shrinks from the right.

/// Fewest / most letters a custom-words term may have to count as an
/// acronym target (letters only — digits and hyphens are ignored).
const ABBREV_MIN_LETTERS: usize = 2;
const ABBREV_MAX_LETTERS: usize = 6;

/// Expander built from the acronym-shaped custom_words.txt terms.
pub struct AbbrevExpander {
    /// (canonical spelling, uppercase letters-only key), e.g. ("TUR-B", "TURB").
    targets: Vec<(String, String)>,
}

impl AbbrevExpander {
    pub fn new(terms: &[String]) -> Self {
        let targets = terms
            .iter()
            .filter(|t| is_abbrev_target(t))
            .map(|t| {
                let key: String = t.chars().filter(|c| c.is_ascii_alphabetic()).collect();
                (t.clone(), key)
            })
            .collect();
        Self { targets }
    }

    pub fn from_corrector(corrector: &WordsCorrector) -> Self {
        let terms: Vec<String> = corrector.plain_terms().iter().map(|s| s.to_string()).collect();
        Self::new(&terms)
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Replace spelled-out letter-name runs with their canonical acronym.
    pub fn expand(&self, text: &str) -> String {
        if self.targets.is_empty() {
            return text.to_string();
        }

        // Tokenize into maximal alphabetic runs with byte ranges.
        let mut tokens: Vec<(usize, usize)> = Vec::new(); // (start, end) byte offsets
        let mut start: Option<usize> = None;
        for (i, c) in text.char_indices() {
            if c.is_alphabetic() {
                if start.is_none() { start = Some(i); }
            } else if let Some(s) = start.take() {
                tokens.push((s, i));
            }
        }
        if let Some(s) = start { tokens.push((s, text.len())); }

        let letters: Vec<Option<char>> = tokens
            .iter()
            .map(|&(s, e)| letter_name(&text[s..e]))
            .collect();

        let mut out = String::with_capacity(text.len());
        let mut cursor = 0; // bytes copied to `out` so far
        let mut i = 0;
        while i < tokens.len() {
            if letters[i].is_none() { i += 1; continue; }
            // Extend the run while the next token is also a letter name and the
            // separator between them is only spaces/hyphens/periods.
            let mut j = i;
            while j + 1 < tokens.len()
                && letters[j + 1].is_some()
                && text[tokens[j].1..tokens[j + 1].0]
                    .chars()
                    .all(|c| c == ' ' || c == '-' || c == '.')
            {
                j += 1;
            }
            // Longest window first; shrink from the right until a target matches.
            let mut matched = false;
            let mut k = j;
            while k > i {
                let key: String = (i..=k).map(|t| letters[t].unwrap()).collect();
                if let Some((canon, _)) = self.targets.iter().find(|(_, tk)| *tk == key) {
                    out.push_str(&text[cursor..tokens[i].0]);
                    out.push_str(canon);
                    cursor = tokens[k].1;
                    i = k + 1;
                    matched = true;
                    break;
                }
                k -= 1;
            }
            if !matched { i += 1; }
        }
        out.push_str(&text[cursor..]);
        out
    }
}

/// A custom-words term is an acronym target if it consists only of
/// letters/digits/hyphens with 2–6 letters, all of them uppercase (MRI, TUR-B).
fn is_abbrev_target(s: &str) -> bool {
    if s.is_empty() || !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return false;
    }
    let letters: Vec<char> = s.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    (ABBREV_MIN_LETTERS..=ABBREV_MAX_LETTERS).contains(&letters.len())
        && letters.iter().all(|c| c.is_ascii_uppercase())
}

/// Map a transcribed token to the letter it names, if any.
/// A single alphabetic character stands for itself ("T U R B" → TURB);
/// otherwise the token must be a German letter name ("em" → M).
fn letter_name(word: &str) -> Option<char> {
    let mut chars = word.chars();
    if let (Some(c), None) = (chars.next(), chars.next()) {
        let u = c.to_ascii_uppercase();
        return u.is_ascii_alphabetic().then_some(u);
    }
    let lower = word.to_lowercase();
    let letter = match lower.as_str() {
        "be" => 'B',
        "ce" => 'C',
        "de" => 'D',
        "ef" | "eff" => 'F',
        "ge" => 'G',
        "ha" => 'H',
        "jot" => 'J',
        "ka" | "kah" => 'K',
        "el" | "ell" => 'L',
        "em" | "emm" => 'M',
        "en" | "enn" => 'N',
        "pe" | "peh" => 'P',
        "ku" => 'Q',
        "er" | "err" => 'R',
        "es" | "ess" => 'S',
        "te" | "teh" => 'T',
        "vau" | "fau" => 'V',
        "we" | "weh" => 'W',
        "ix" | "iks" => 'X',
        "ypsilon" | "ipsilon" | "üpsilon" => 'Y',
        "zett" | "zet" => 'Z',
        _ => return None,
    };
    Some(letter)
}

/// Drop a single leading `0` (an edge-vowel code) so that vowel-initial and
/// consonant-initial renderings of the same word compare equal under the
/// phonetic gate (e.g. `01264` "Epetmika" vs `1264` "Betmiga").
fn strip_leading_zero(code: &str) -> &str {
    code.strip_prefix('0').unwrap_or(code)
}

/// A term is fuzzy-matchable only if it is a single all-alphabetic word.
fn is_single_word(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphabetic())
}

/// Compute the Kölner Phonetik (Cologne phonetics) code of a word.
///
/// Cologne phonetics is the German-language counterpart to Soundex: it collapses
/// phonetically equivalent letters (t↔d, g↔k↔q, s↔z, …) to the same digit, so
/// "Toviaz"/"Tovias" → `238` and "Betmiga"/"Bedmika" → `1264`.
///
/// Algorithm: context-sensitive per-letter digit coding, then collapse adjacent
/// duplicate digits, then drop all `0` codes except a leading one.
fn koelner_phonetik(s: &str) -> String {
    let letters = normalize_letters(s);
    if letters.is_empty() {
        return String::new();
    }

    let mut raw: Vec<char> = Vec::with_capacity(letters.len() + 2);
    for i in 0..letters.len() {
        let cur = letters[i];
        let prev = if i > 0 { Some(letters[i - 1]) } else { None };
        let next = letters.get(i + 1).copied();
        match cur {
            'A' | 'E' | 'I' | 'J' | 'O' | 'U' | 'Y' => raw.push('0'),
            'H' => {} // not coded
            'B' => raw.push('1'),
            'P' => raw.push(if next == Some('H') { '3' } else { '1' }),
            'D' | 'T' => {
                raw.push(if matches!(next, Some('C') | Some('S') | Some('Z')) { '8' } else { '2' })
            }
            'F' | 'V' | 'W' => raw.push('3'),
            'G' | 'K' | 'Q' => raw.push('4'),
            'L' => raw.push('5'),
            'M' | 'N' => raw.push('6'),
            'R' => raw.push('7'),
            'S' | 'Z' => raw.push('8'),
            'C' => {
                let code = if prev.is_none() {
                    // Word-initial C.
                    if matches!(next, Some('A') | Some('H') | Some('K') | Some('L')
                        | Some('O') | Some('Q') | Some('R') | Some('U') | Some('X')) { '4' } else { '8' }
                } else if matches!(prev, Some('S') | Some('Z')) {
                    '8'
                } else if matches!(next, Some('A') | Some('H') | Some('K') | Some('O')
                    | Some('Q') | Some('U') | Some('X')) {
                    '4'
                } else {
                    '8'
                };
                raw.push(code);
            }
            'X' => {
                if matches!(prev, Some('C') | Some('K') | Some('Q')) {
                    raw.push('8');
                } else {
                    raw.push('4');
                    raw.push('8');
                }
            }
            _ => {} // non-letter (shouldn't occur after normalization)
        }
    }

    // Collapse adjacent duplicate digits.
    let mut collapsed: Vec<char> = Vec::with_capacity(raw.len());
    for d in raw {
        if collapsed.last() != Some(&d) {
            collapsed.push(d);
        }
    }

    // Drop all '0' codes except a leading one.
    let mut out = String::with_capacity(collapsed.len());
    for (i, d) in collapsed.iter().enumerate() {
        if *d != '0' || i == 0 {
            out.push(*d);
        }
    }
    out
}

/// Uppercase, map German umlauts/ß to base letters, drop non-letters.
fn normalize_letters(s: &str) -> Vec<char> {
    let mut out = Vec::new();
    for c in s.chars() {
        match c {
            'ä' | 'Ä' => out.push('A'),
            'ö' | 'Ö' => out.push('O'),
            'ü' | 'Ü' => out.push('U'),
            'ß' => { out.push('S'); out.push('S'); }
            _ => {
                if let Some(u) = c.to_uppercase().next() {
                    if u.is_ascii_alphabetic() {
                        out.push(u);
                    }
                }
            }
        }
    }
    out
}

/// Classic Levenshtein edit distance (insert/delete/substitute = 1).
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() { return b.len(); }
    if b.is_empty() { return a.len(); }

    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1)
                .min(cur[j] + 1)
                .min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn koelner_phonetic_equivalence() {
        // The exact misspellings the user reported must share the target's code.
        assert_eq!(koelner_phonetik("Betmiga"), koelner_phonetik("Bedmika"));
        assert_eq!(koelner_phonetik("Toviaz"), koelner_phonetik("Tovias"));
        // Known reference codes.
        assert_eq!(koelner_phonetik("Betmiga"), "1264");
        assert_eq!(koelner_phonetik("Toviaz"), "238");
    }

    #[test]
    fn koelner_handles_umlauts() {
        assert_eq!(koelner_phonetik("Müller"), koelner_phonetik("Mueller"));
    }

    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein("toviaz", "tovias"), 1);
        assert_eq!(levenshtein("betmiga", "bedmika"), 2);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
    }

    #[test]
    fn fuzzy_snaps_reported_misspellings() {
        let terms = vec!["Betmiga".to_string(), "Toviaz".to_string()];
        let m = FuzzyMatcher::new(&terms);
        assert_eq!(m.correct("Patient nimmt Bedmika und Tovias.", 0.34),
                   "Patient nimmt Betmiga und Toviaz.");
    }

    #[test]
    fn fuzzy_snaps_leading_vowel_drift() {
        // Voxtral (no biasing) prepends an edge vowel: "Epetmika" → code 01264,
        // "Betmiga" → 1264. Identical-code matching missed this; the relaxed gate must catch it.
        let terms = vec!["Betmiga".to_string()];
        let m = FuzzyMatcher::new(&terms);
        assert_eq!(koelner_phonetik("Epetmika"), "01264");
        assert_eq!(m.correct("Patient nimmt Epetmika.", 0.5), "Patient nimmt Betmiga.");
    }

    #[test]
    fn fuzzy_preserves_unrelated_words() {
        let terms = vec!["Betmiga".to_string()];
        let m = FuzzyMatcher::new(&terms);
        // "Migration" is phonetically far from "Betmiga" — must not be touched.
        let input = "Die Migration verlief ohne Probleme.";
        assert_eq!(m.correct(input, 0.34), input);
    }

    #[test]
    fn fuzzy_leaves_exact_match_unchanged() {
        let terms = vec!["Toviaz".to_string()];
        let m = FuzzyMatcher::new(&terms);
        assert_eq!(m.correct("Toviaz", 0.34), "Toviaz");
    }

    #[test]
    fn fuzzy_skips_short_words() {
        let terms = vec!["Abca".to_string()];
        let m = FuzzyMatcher::new(&terms);
        // "Abc" is shorter than FUZZY_MIN_LEN, never matched.
        assert_eq!(m.correct("Abc", 0.5), "Abc");
    }

    #[test]
    fn fuzzy_excludes_multiword_and_hyphenated_targets() {
        let terms = vec!["TUR-B".to_string(), "von Willebrand".to_string()];
        let m = FuzzyMatcher::new(&terms);
        assert!(m.is_empty());
    }

    #[test]
    fn fuzzy_targets_from_plain_terms_only() {
        // Replacement pairs (with '=') must not become fuzzy targets.
        let c = WordsCorrector::from_str("Migration=Miktion\nBetmiga\nToviaz\n");
        let m = FuzzyMatcher::from_corrector(&c);
        assert_eq!(m.correct("Bedmika", 0.34), "Betmiga");
    }

    #[test]
    fn abbrev_target_detection() {
        assert!(is_abbrev_target("MRI"));
        assert!(is_abbrev_target("TUR-B"));
        assert!(is_abbrev_target("PSA"));
        assert!(!is_abbrev_target("Betmiga"));     // not uppercase
        assert!(!is_abbrev_target("M"));           // too short
        assert!(!is_abbrev_target("LANGWORT"));    // too long
        assert!(!is_abbrev_target("von Willebrand")); // space
    }

    #[test]
    fn abbrev_expands_letter_name_runs() {
        let c = WordsCorrector::from_str("MRI\nTUR-B\nPSA\nEKG\nBetmiga\n");
        let a = AbbrevExpander::from_corrector(&c);
        assert_eq!(a.expand("Ein Em Er I wurde durchgeführt."),
                   "Ein MRI wurde durchgeführt.");
        assert_eq!(a.expand("Zustand nach Te U Er Be im März."),
                   "Zustand nach TUR-B im März.");
        assert_eq!(a.expand("Der Pe Es A Wert ist stabil."),
                   "Der PSA Wert ist stabil.");
        // Hyphen-joined rendering of the same dictation
        assert_eq!(a.expand("EM-ER-I unauffällig."), "MRI unauffällig.");
        // Single capital letters spell themselves
        assert_eq!(a.expand("E Ka Ge ohne Befund."), "EKG ohne Befund.");
    }

    #[test]
    fn abbrev_leaves_normal_text_alone() {
        let c = WordsCorrector::from_str("MRI\nPSA\n");
        let a = AbbrevExpander::from_corrector(&c);
        // "er"/"es" are letter names but the runs don't match any target.
        let input = "Er sagt, es geht ihm gut.";
        assert_eq!(a.expand(input), input);
        // Already-canonical acronym is a single token — never a ≥2-token run.
        assert_eq!(a.expand("Das MRI war unauffällig."), "Das MRI war unauffällig.");
    }

    #[test]
    fn abbrev_shrinks_window_to_match() {
        let c = WordsCorrector::from_str("MRI\n");
        let a = AbbrevExpander::from_corrector(&c);
        // "u" extends the letter-name run (MRIU matches nothing) — the window
        // must shrink from the right and still find MRI.
        assert_eq!(a.expand("Em Er I u Schall."), "MRI u Schall.");
    }
}
