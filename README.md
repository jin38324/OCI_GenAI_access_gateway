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

# Quick Start

1. Clone this repository and set prerequisites;

2. Run this app:

    ```bash
    uvicorn api.app:app --host 0.0.0.0 --port 8088 --reload
    ```

3. Config your application like this:
![alt text](image/setting.png)

It's OK now!

![alt text](image/chat.png)



# Prerequisites

1. In this project, we use OCI python SDK to call cloud services. 

    1.1 `pip install oci`

    1.2 create config file on OCI console, follow this [SDK and CLI Configuration File](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm)

    1.3 Notice that we add `compartment_id` in config file.

2. You can modify the `api/setting.py` file to custom config file location and DEFAULT_API_KEYS.

3. It's done. You can edit other settings if you want.

# Models
List of OCI Generative AI Service models currently supported:

**Chat models**:
- meta.llama-3-70b-instruct
- cohere.command-r-plus
- cohere.command-r-16k

**Embedding models**:
- cohere.embed-english-v3.0
- cohere.embed-multilingual-v3.0
- cohere.embed-english-light-v3.0
- cohere.embed-multilingual-light-v3.0

# Features under development

- Tool call


# Cahnge log
- 20240729: first commit;
