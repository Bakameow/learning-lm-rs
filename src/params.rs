use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use half::bf16;

pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        //let names = safetensor.names();
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor_view = safetensor.tensor(name).expect(&format!("tensor {name} not found"));
            let shape = tensor_view.shape().to_vec();
            let data = tensor_view.data().chunks_exact(4)
                .map(|chunk|{
                    let array:[u8;4] = chunk.try_into().unwrap();
                    f32::from_le_bytes(array)
                })
                .collect();
            Tensor::new(data, &shape)
        };
        let n_layers = config.num_hidden_layers;
        LLamaParams {
            embedding_table: if config.tie_word_embeddings {
                    get_tensor("lm_head.weight")
                } else {
                    get_tensor("model.embed_tokens.weight")
                },
            rms_att_w: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.input_layernorm.weight"))).collect(),
            wq: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.self_attn.q_proj.weight"))).collect(),
            wk: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.self_attn.k_proj.weight"))).collect(),
            wv: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.self_attn.v_proj.weight"))).collect(),
            wo: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.self_attn.o_proj.weight"))).collect(),
            rms_ffn_w: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.post_attention_layernorm.weight"))).collect(),
            w_up: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.mlp.up_proj.weight"))).collect(),
            w_gate: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.mlp.gate_proj.weight"))).collect(),
            w_down: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.mlp.down_proj.weight"))).collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}

impl LLamaParams<bf16> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        //let names = safetensor.names();
        let get_tensor = |name: &str| -> Tensor<bf16> {
            let tensor_view = safetensor.tensor(name).expect(&format!("tensor {name} not found"));
            let shape = tensor_view.shape().to_vec();
            let data = tensor_view.data().chunks_exact(2)
                .map(|chunk|{
                    let array:[u8;2] = chunk.try_into().unwrap();
                    bf16::from_le_bytes(array)
                })
                .collect();
            Tensor::new(data, &shape)
        };
        let n_layers = config.num_hidden_layers;
        LLamaParams {
            embedding_table: if config.tie_word_embeddings {
                    get_tensor("lm_head.weight")
                } else {
                    get_tensor("model.embed_tokens.weight")
                },
            rms_att_w: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.input_layernorm.weight"))).collect(),
            wq: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.self_attn.q_proj.weight"))).collect(),
            wk: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.self_attn.k_proj.weight"))).collect(),
            wv: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.self_attn.v_proj.weight"))).collect(),
            wo: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.self_attn.o_proj.weight"))).collect(),
            rms_ffn_w: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.post_attention_layernorm.weight"))).collect(),
            w_up: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.mlp.up_proj.weight"))).collect(),
            w_gate: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.mlp.gate_proj.weight"))).collect(),
            w_down: (0..n_layers).map(|i| get_tensor(&format!("model.layers.{i}.mlp.down_proj.weight"))).collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
