mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use actix_web::{get, post, App, HttpResponse, HttpServer, Responder};

#[derive(Serialize,Deserialize)]
struct Request {
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

#[post("/chat")]
async fn chat(request: String) -> impl Responder {
    let prompt_json : Request = serde_json::from_str(&request).expect("Deserialize Prompt failed");
    let project_dir = env!("CARGO_MANIFEST_DIR"); 
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = format!("<|im_start|>system\n{0}<|im_end|>\n<|im_start|>user\n{1}<|im_end|>\n<|im_start|>assistant",prompt_json.system_message,prompt_json.user_message);
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
    let ans = tokenizer.decode(&output_ids, true).unwrap();
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

//const DIR : &str = "chat";
//
//fn main(){
//    let project_dir = env!("CARGO_MANIFEST_DIR");
//    let model_dir = PathBuf::from(project_dir).join("models").join(DIR);
//    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
//    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
//    let mut input = format!("<|im_start|>system\n{0}<|im_end|>\n<|im_start|>user\n{1}<|im_end|>\n<|im_start|>assistant",&"you are a helpfull assistant",&"it is rainy.should i bring a umbralla with me?");
//        if DIR == "story"{
//        input = "Once upon a time".to_string();
//    }
//    println!("prompt = {}",&input);
//    let binding = tokenizer.encode(input.clone(), true).unwrap();
//    let input_ids = binding.get_ids();
//    let output_ids = llama.generate(
//        input_ids,
//        500,
//        0.8,
//        30,
//        1.0,
//    );
//    let ans = tokenizer.decode(&output_ids, true).unwrap();
//    println!("{}",ans);
//}
