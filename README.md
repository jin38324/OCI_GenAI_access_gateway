**Oracle Cloud Infrastructure (OCI) Generative AI Service** is a fully managed service that integrates these versatile language models into a variety of use cases.

Oracle has released SDK that makes it easy to call OCI Generative AI services. However, for many packaged projects, some code modification is required to integrate the OCI Generative AI services.

Due to the wide application of OpenAI services, its API interface format has been supported by the vast majority of AI applications. In order to speed up application integration, a project has been created to make OCI Generative AI services compatible with the OpenAI API.

With this project, you can quickly integrate any application that supports a custom OpenAI interface without modifying the application.

---

**Oracle 云基础设施 (OCI) 生成式 AI** 是一种完全托管的服务，可将这些多功能语言模型集成到各种用例中。

Oracle已经发布了SDK，可以方便地调用OCI生成式AI服务。但是对于很多已经包装好的项目，需要一些代码修改工作量，以集成OCI上的生成式AI服务。

由于OpenAI服务的广泛应用，其API接口格式已经被绝大多数AI应用所支持。为了能够加快应用集成，一个使OCI生成式AI服务兼容OpenAI API的项目被创建。

通过此项目，您可以在不修改应用的情况下，快速集成任何支持自定义OpenAI接口的应用程序。

*This is a project inspired by [aws-samples/bedrock-access-gateway](https://github.com/aws-samples/bedrock-access-gateway/tree/main)*

# Change log
- 20241031: Add MIT license
- 20241022: Support LLM service deployed through the AI ​​Quick Action of OCI Data Science; Optimize model configuration;
- 20240905: Support Instance principals auth;
- 20240815: Add Dedicated AI Cluster support;
- 20240729: first commit;

# Quick Start


1. Clone this repository and [set prerequisites](#set-prerequisites);

2. Run this app:
    2.1 Launch on host

    ```bash
    python app.py
    ```
    2.1 Launch in docker

    copy `.oci` directory to `/root`, and confirm the `key_file` parameter is set to `/root` directory.
    ```bash
    docker build -t oci_genai_gateway .

    docker run -p 8088:8088 \
            -v $(pwd)/app/config.py:/app/config.py \
            -v $(pwd)/app/models.yaml:/app/models.yaml \
            -v /root/.oci:/root/.oci \
            -it oci_genai_gateway

    ```

3. Config your application like this:
![alt text](image/setting.png)

It's OK now!

![alt text](image/chat.png)


# Set Prerequisites

## 1. Install python packages
`pip install -r requirements.txt`

## 2. Set authentication
Create access authentication for OCI. There are two ways to achieve this:
- Use API Key. This need a little effort, but easy to understand and can be used everywhere
- Use instance principal. This is easy to set but only available on OCI host machines.

**Option1: Use API Key**    

create config file on OCI console, follow this [SDK and CLI Configuration File](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm).

Set `config.py` file, point to your config location,like `OCI_CONFIG_FILE = "~/.oci/config"`.

**Option2: Use instance principal setting**

Set [OCI policy](https://docs.oracle.com/en-us/iaas/Content/Identity/policieshow/Policy_Basics.htm), define
```
allow dynamic-group <xxxx> to manage generative-ai-family in tenancy
```
xxxx is your dynamic-group that indicated your vm or other resources

in `config.py`, set `AUTH_TYPE=INSTANCE_PRINCIPAL`

## Other settings:

You can modify the `config.py` file to change default settings.
- `PORT`: service http port
- `RELOAD`: if True, the web service will reload if any file change in the project
- `DEBUG`: if True, more logs will displayed
- `DEFAULT_API_KEYS`: Authorize token for the API, default is `ocigenerativeai`
- `API_ROUTE_PREFIX`: API url PREDIX
- `AUTH_TYPE`: `API_KEY` or `INSTANCE_PRINCIPAL`
- `OCI_CONFIG_FILE`: OCI config file location, default is `~/.oci/config`
- `OCI_CONFIG_FILE_KEY`: multiple configs can be added in one config file, so you can use key to determain use which one
- `INFERENCE_ENDPOINT_TEMPLATE`: no need to modify, unless the OCI service changes


# Models

Generative AI is a rapidly evolving field, with new models being added and old models being retired, so we gave up hard-coding model information in the code and instead defined the model through `models.yaml`.

Don't worry, most of the models have been written well in the file, you just need to use them.

You can define 4 types of models:
- **ondemand**: pre-trained chat model provided by OCI generative AI service, accessed through a unified API.
- **embedding**: pre-trained embedding model provided by OCI generative AI service, accessed through a unified API.
- **dedicated**: OCI Generative AI service’s proprietary model, the model to be accessed is determined by specifying the endpoint
- **datascience**: LLM service deployed through the [AI ​​Quick Action function of OCI Data Science](https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions.htm).
AI Quick Actions makes it easy for you to browse foundation models, and deploy, fine-tune, and evaluate them inside Data Science notebooks.
The model to be accessed is determined by specifying the endpoint, and endpoint should be end with `/predict`. 
When create datascience deployment `Inference mode`should be `/v1/chat/completions`.`Inference container` should be `VLLM`.

Model information parameters:
- `region`: OCI services can be provided by multiple regions, so you can configure the region to be called
- `compartment_id`: Required, this parameter determines the compartment where the service is initiated, which is basically related to cost and permission management;
- `name`: is a custom name, a legal string is fine
- `model_id`: is the [standard model ID](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm)
- `endpoint`: Call endpoint, which can be viewed through the OCI console


# Test the application

## Set base_url

```python
from openai import OpenAI

client = OpenAI(
    api_key = "ocigenerativeai",
    base_url = "http://xxx.xxx.xxx.xxx:8088/api/v1/",
    )
models = client.models.list()
for model in models:
    print(model.id)
```

## Chat 
```python

test_models = [
    "cohere.command-r-plus",
    "cohere.command-r-16k",
    "meta.llama-3.1-70b-instruct",
    "meta.llama-3.1-405b-instruct",
    "ODSC-Mistral-7B-Instruct"]

message = "Hello!"

# Test chat completions
for model_name in test_models:
    print("Model:",model_name,)
    print("User:", message)
    completion = client.chat.completions.create(
        model = model_name,
        messages = [{"role": "user", "content": message}],
        max_tokens=12,
        )
    print("Assistant:", completion.choices[0].message.content)
    print("*"*100)
```
output:
```
Model: cohere.command-r-plus
User: Hello!
Assistant: Hello to you too
**************************************************
Model: cohere.command-r-16k
User: Hello!
Assistant: Hello there! How's it going? I'm an AI
**************************************************
Model: meta.llama-3.1-70b-instruct
User: Hello!
Assistant: Hello! How can I assist you today?
**************************************************
Model: meta.llama-3.1-405b-instruct
User: Hello!
Assistant: Hello! It's nice to meet you. Is there something
**************************************************
Model: ODSC-Mistral-7B-Instruct
User: Hello!
Assistant: Hi there! How can I help you today? If you
**************************************************
```

## chat streaming
```python
test_models = [
    "cohere.command-r-plus",
    "cohere.command-r-16k",
    "meta.llama-3.1-70b-instruct",
    "meta.llama-3.1-405b-instruct",
    "ODSC-Mistral-7B-Instruct"]
# Test chat completions with streaming response
for model in test_models:
    print("Model:", model)
    print("User:", message)
    print("Assistant:", end='')
    response = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': message}],
        max_tokens=12,
        stream=True  # this time, we set stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content,end='')
    print('\n',"*"*100)
```

output:
```
Model: cohere.command-r-plus
User: Hello!
Assistant:Hello! What can I do for you today?
 **************************************************
Model: cohere.command-r-16k
User: Hello!
Assistant:Hello! It's nice to receive a message from you.
 **************************************************
Model: meta.llama-3.1-70b-instruct
User: Hello!
Assistant:Hello! It's nice to meet you. Is there
 **************************************************
Model: meta.llama-3.1-405b-instruct
User: Hello!
Assistant:Hello! How can I help you today?
 **************************************************
Model: ODSC-Mistral-7B-Instruct
User: Hello!
Assistant: Hello! How can I help you today? I'm
 **************************************************
```

## embedding
```python
embd_model=[ 'cohere.embed-english-light-v3.0',
            'cohere.embed-english-v3.0',
            'cohere.embed-multilingual-light-v3.0',
            'cohere.embed-multilingual-v3.0']
input = ["hello!","你好！"]
for model in embd_model:
    print("model:",model)
    response = client.embeddings.create(input = input, model=model).data
    for each in response:
        print(each.index,':',str(each.embedding)[:36],'......',str(each.embedding)[-36:])
    print("*"*100)
```

output:
```
model: cohere.embed-english-light-v3.0
0 : [0.049652099609375, 0.03176879882812 ...... 0823211669921875, 0.021026611328125]
1 : [-0.0679931640625, -0.05831909179687 ...... 0200042724609375, 0.040069580078125]
****************************************************************************************************
model: cohere.embed-english-v3.0
0 : [-0.016204833984375, 0.0127410888671 ...... 0.02081298828125, 0.020172119140625]
1 : [-0.01422882080078125, -0.0110321044 ...... 021820068359375, 0.0208587646484375]
****************************************************************************************************
model: cohere.embed-multilingual-light-v3.0
0 : [-0.0188446044921875, 0.032745361328 ...... .039581298828125, 0.000946044921875]
1 : [0.03814697265625, 0.013946533203125 ...... 0.08721923828125, 0.029815673828125]
****************************************************************************************************
model: cohere.embed-multilingual-v3.0
0 : [-0.006015777587890625, 0.0308074951 ...... -0.00399017333984375, -0.0185546875]
1 : [-0.00856781005859375, 0.04287719726 ...... 9579658508300781, -0.04571533203125]
****************************************************************************************************
```


# Features under development
- Tool call
- Docker deployment
- Multi-modal 
