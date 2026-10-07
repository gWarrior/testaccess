//! Does the SNN memory find "Моего друга зовут N. N - мой друг." amid
//! prose when asked "Кто такой N -"? The full 9-token probe against the
//! current-line probe (`model::probe_of`). Needs the prepared corpus:
//! `cargo run --release -p snn-lm --example probe_line`.
use snn_memory::{ContextConfig, ContextMemory, Probe};
fn main() {
    let tok =
        snn_lm::tokenizer::Tokenizer::load(std::fs::File::open("/home/user/data/prepared/tokenizer.bpe").unwrap())
            .unwrap();
    let f = snn_lm::data::TokenFile::load(std::path::Path::new("/home/user/data/prepared/val.bin")).unwrap();
    let prose: Vec<u32> = f.tokens[..60_000].iter().map(|&t| t as u32).collect();
    let (mut hit_full, mut hit_line, mut n) = (0, 0, 0);
    for (k, name) in ["Лами", "Роке", "Нита", "Кавен", "Тумио", "Серал", "Бирта", "Олен", "Мирад"].iter().enumerate()
    {
        let stmt = tok.encode(&format!("\nМоего друга зовут {name}. {name} - мой друг.\n"));
        let q = tok.encode(&format!("и пошёл дальше.\nКто такой {name} -"));
        let mut ctx = ContextMemory::new(ContextConfig { max_tokens: 177_147, ..Default::default() }).unwrap();
        let off = 3000 + k * 5000;
        ctx.append(&prose[..off]).unwrap();
        let s0 = ctx.position();
        ctx.append(&stmt).unwrap();
        ctx.append(&prose[off..off + 19_683.min(prose.len() - off)]).unwrap();
        let s1 = s0 + stmt.len() as u64;
        let recent = &q[q.len() - 9..];
        for (probe, hit) in [(recent, &mut hit_full), (snn_lm::model::probe_of(recent), &mut hit_line)] {
            let r = ctx.retrieve(Probe::Tokens(probe), 3).unwrap();
            if r.spans.iter().any(|s| s.start < s1 && s.end() > s0) {
                *hit += 1;
            }
        }
        n += 1;
    }
    println!("statement retrieved: full 9-token probe {hit_full}/{n}, line probe {hit_line}/{n}");
}
