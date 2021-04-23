extern crate tokenizers as tk;
use tk::models::wordpiece::WordPiece;
use tk::Tokenizer;
use tk::tokenizer::{AddedToken};
use tk::pre_tokenizers::whitespace::Whitespace;
use tk::decoders::wordpiece::WordPiece as WordPieceDecoder;

fn main() {
    println!("Hello, world!");
    let wp = WordPiece::builder()
        .files(String::from("./vocab.txt"))
        .max_input_chars_per_word(100)
        .build()
        .expect("Fail to build wp");
    let mut tokenizer = Tokenizer::new(wp);
    tokenizer
        .with_pre_tokenizer(Whitespace::default())
        .with_decoder(WordPieceDecoder::default());

    // let sts = "special_tokens.txt";
    // let fp_sts = File::open(sts).expect("Failed to open the file");
    // let reader = io::BufReader::new(fp_sts);
    // let mut tokens: Vec<AddedToken> = Vec::new();
    // for line in reader.lines() {
    //     match line {
    //         Ok(l) => {
    //             println!("special token {}", l);
    //             tokens.push(AddedToken::from(String::from(l.trim()), true).single_word(true));
    //         },
    //         Err(_) => println!("Error when reading the line"),
    //     }
    // }
    // tokenizer.add_special_tokens(&tokens[..]);

    let special_tokens_str = "[UNK] [SEP] [PAD] [CLS] [MASK]";
    let special_tokens = special_tokens_str.split(" ");
    let mut tokens: Vec<AddedToken> = Vec::new();
    for token in special_tokens {
        tokens.push(AddedToken::from(String::from(token), true).single_word(true));
    }
    tokenizer.add_special_tokens(&tokens[..]);

    let line = String::from("In sleep he sang to me, in [MASK] he came.");
    let encoded = tokenizer.encode(line.clone(), false).expect("Failed to encode");

    println!("Input: {}", line);
    println!("Tokens:\t\t{:?}", encoded.get_tokens());
    println!("IDs:\t\t{:?}", encoded.get_ids());
    println!("Offsets:\t{:?}", encoded.get_offsets());
    println!(
        "Decoded:\t{}",
        tokenizer.decode(encoded.get_ids().to_vec(), true).unwrap()
    );
    // don't know how to use it yet.
}
