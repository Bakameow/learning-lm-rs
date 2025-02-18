mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::fs::File;
use std::{f32, path::PathBuf};
use half::bf16;
use operators::ToF32;
use params::Load;
use serde::{Deserialize, Serialize};
use crate::config::LlamaConfigJson;

use tokenizers::Tokenizer;
use actix_web::{get, post, App, HttpResponse, HttpServer, Responder};

#[derive(Serialize,Deserialize)]
struct Request {
    history: String,
    system_message: String,
    user_message: String,
}

#[get("/story")]
async fn story() -> impl Responder {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    let output_ids = llama.generate(
        input_ids,
        200,
        0.8,
        30,
        0.6,
    );
    let mut ans = tokenizer.decode(&output_ids, true).unwrap();
    ans.insert_str(0,input);
    HttpResponse::Ok().body(ans)
}

fn chat_func<T>(model_dir: PathBuf,prompt: Request) -> String 
where T: Default + Copy +Load + ToF32
{
    let llama = model::Llama::<T>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = format!("{0}<|im_start|>system\n{1}<|im_end|>\n<|im_start|>user\n{2}<|im_end|>\n<|im_start|>assistant",
                        prompt.history,prompt.system_message,prompt.user_message);
    println!("{}",&prompt.history);
    println!("{}",&input);
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    let output_ids = llama.generate(
        input_ids,
        200,
        0.8,
        30,
        1.,
    );
    tokenizer.decode(&output_ids, true).unwrap()
}

#[post("/chat")]
async fn chat(request: String) -> impl Responder {
    let prompt_json : Request = serde_json::from_str(&request).expect("Deserialize Prompt failed");
    let project_dir = env!("CARGO_MANIFEST_DIR"); 
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let config = File::open(model_dir.join("config.json")).unwrap();
    let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
    let ans = match config.torch_dtype.as_ref() {
        "bfloat16" => chat_func::<bf16>(model_dir, prompt_json),
        "float32" => chat_func::<f32>(model_dir, prompt_json),
        _ => todo!()
    };
    HttpResponse::Ok().body(ans)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Server running on http://127.0.0.1:8080");
    HttpServer::new(|| {
        App::new()
            .service(story)
            .service(chat)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

#[test]
fn infer_test(){
    let dir = "chat";
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join(dir);
    let config = File::open(model_dir.join("config.json")).unwrap();
    let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
    let prompt_json = Request {history:"".to_string(),system_message:"you are a helpful assistant".to_string(),user_message:"who are you?".to_string()};
    let ans = match config.torch_dtype.as_ref() {
        "bfloat16" => chat_func::<bf16>(model_dir, prompt_json),
        "float32" => chat_func::<f32>(model_dir, prompt_json),
        _ => todo!()
    };
    println!("{}",ans);
}
