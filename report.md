# 大模型推理系统项目报告

## 基本指标
1. 实现GQA，在计算attention矩阵和V矩阵时引入了转置，以提高行优先访存局部性
2. 成功加载char模型进行对话

```bash
cargo run
```
将在本地的8080端口开启服务，可以通过发送get请求到`/story`完成故事续写，或发送post请求到`/chat`进行对话

```bash
#!/bin/zsh
# URL
url="http://localhost:8080/chat"
# JSON 数据
json_data='{"session_id":"*****","history":"","system_message":"you are a helpful assistant.", "user_message":"one plus one equal to?"}'
# 发送 POST 请求
curl -X POST -H "Content-Type: application/json" -d "$json_data" "$url"
```

## 拓展指标
1. 支持W16A32推理，运行时反量化实现混合精度推理
2. 基于actix实现API访问

## 后续计划
1. 进一步支持int8、int4的量化推理
2. 后端实现多会话管理
3. FFN支持CUDA加速
