extern crate tokenizers as tk;
use tk::models::wordpiece::WordPiece;
use tk::tokenizer::AddedToken;
use tk::Tokenizer;
use tk::pre_tokenizers::whitespace::Whitespace;
use tk::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tk::normalizers::bert::BertNormalizer;
use std::ffi::{CStr};
use std::os::raw::{c_char, c_int};
use std::string::String;

static mut TOKENIZERS: Vec<Tokenizer> = Vec::new();
static mut NUM_CONTEXTS: i32 = 0;

#[repr(C)]
pub struct Query {
    input: *const c_char,
    input_len: c_int,
    token_ids: *mut c_int,
    num_ids: c_int,
    tokenize_ok: c_int,
}

impl Query {
    fn get_input_string(&self) -> String {
        charp_to_str(self.input)
    }

    fn set_token_ids(&mut self, ids: &[u32], max_length: usize) {
        unsafe {
            let arr = std::slice::from_raw_parts_mut(self.token_ids, max_length);
            for (i, id) in ids.iter().enumerate() {
                if i < max_length {
                    arr[i] = id.to_owned() as i32;
                } else {
                    break
                }
            }
            self.num_ids = std::cmp::min(max_length, ids.len()) as i32;
        }
    }

    fn set_flag(&mut self, ok: bool) {
        self.tokenize_ok = match ok {
            true => 1,
            false => 0,
        };
    }
}

fn charp_to_str(charp: *const c_char) -> String {
    unsafe {
        let line = CStr::from_ptr(charp);
        let slice = line.to_str().unwrap();
        String::from(slice)
    }
}

fn get_tokens_from_str(token_str: &str) -> Vec<AddedToken> {
    let mut ret: Vec<AddedToken> = Vec::new();

    for token in token_str.split(" ") {
        ret.push(
            AddedToken::from(
                String::from(token),
                true
            ).single_word(true)
        );
    }
    ret
}

fn acquire_tokenizer() -> &'static tk::Tokenizer {
    unsafe {
        match TOKENIZERS.len() {
            0 => panic!("Tokenizer must be initialized before use."),
            size => &TOKENIZERS[size-1],
        }
    }
}

fn process_one_query(
    tokenizer: &tk::Tokenizer,
    query: &mut Query,
    max_num_ids: c_int,
) {
    let line = query.get_input_string();
    let encoded = tokenizer.encode(line.to_owned(), false).expect("Encode failed!");
    query.set_token_ids(encoded.get_ids(), max_num_ids as usize);
    query.set_flag(true);
}

#[no_mangle]
pub extern "C" fn BertTokenizerV2Init(
    p_vocab_file: *const c_char,
    p_special_tokens: *const c_char,
    do_lower_case: bool,
    max_input_chars_per_word: c_int,
    num_contexts: c_int,
) -> bool {
    let vocab_file = charp_to_str(p_vocab_file);
    let wp_builder = WordPiece::builder();
    let configured_builder = wp_builder
        .files(vocab_file)
        .max_input_chars_per_word(max_input_chars_per_word as usize);
    let wp = match configured_builder.build() {
        Ok(m) => m,
        Err(_) => return false
    };
    let special_tokens = get_tokens_from_str(charp_to_str(p_special_tokens).as_str());
    let normalizer = BertNormalizer::new(true, true, Some(true), do_lower_case);

    let mut tokenizer = Tokenizer::new(wp);

    tokenizer
        .with_normalizer(normalizer)
        .with_pre_tokenizer(Whitespace::default())
        .with_decoder(WordPieceDecoder::default())
        .add_special_tokens(&special_tokens[..]);

    unsafe {
        NUM_CONTEXTS = num_contexts;
        TOKENIZERS.push(tokenizer);
    }

    true
}

#[no_mangle]
pub extern "C" fn BertTokenizerV2Tokenize(
    query: *mut Query,
    max_num_ids: c_int,
) {
    let tokenizer = acquire_tokenizer();
    unsafe {
        process_one_query(tokenizer, &mut *query, max_num_ids);
    }
}

#[no_mangle]
pub extern "C" fn BertTokenizerV2BatchTokenize(
    batch_size: c_int,
    queries: *mut Query,
    max_num_ids: c_int,
) {
    let tokenizer = acquire_tokenizer();
    unsafe {
        let arr = std::slice::from_raw_parts_mut(queries, batch_size as usize);
        for query in arr.iter_mut() {
            process_one_query(&tokenizer, &mut *query, max_num_ids);
        }
    }
}